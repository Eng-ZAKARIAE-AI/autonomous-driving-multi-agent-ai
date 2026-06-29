"""Dashboard WebSocket server — découplé de la boucle CARLA.

Principe
--------
La boucle CARLA appelle push() à chaque tick (fire-and-forget, jamais bloquant).
Le serveur WebSocket tourne dans un thread daemon séparé avec son propre event loop asyncio.
Si aucun client n'est connecté, ou si le client est lent → les états sont drainés
silencieusement ; la simulation n'est jamais impactée.

Pattern "latest-value" : le broadcaster draine toute la queue à chaque itération
et n'envoie que le dernier état. Ainsi, même si push() est appelé 20 fois/s et que
le client est plus lent, on ne consomme pas de mémoire indéfiniment.

Robustesse déconnexion
----------------------
ping_interval=None désactive le mécanisme keepalive websockets (inutile sur localhost
et source de faux positifs avec asyncio en thread daemon sous Python 3.7/Windows).
Un client qui se déconnecte est retiré de _clients via _handler (wait_closed) ET via
le dead-set du broadcaster (exception sur ws.send).

Shutdown propre
---------------
stop() positionne self._running=False. Le broadcaster sort de sa boucle while à la
prochaine itération (≤50ms), ce qui termine _serve() et ferme websockets.serve
proprement. Aucune erreur asyncio.
"""

import asyncio
import json
import math
import queue
import threading
import logging
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)


def _sanitize(obj: Any) -> Any:
    """Remplace récursivement inf/-inf/nan par None avant json.dumps.

    json.dumps(float('inf')) produit 'Infinity' (JSON invalide) sans lever
    d'exception. JSON.parse() côté browser lève SyntaxError. Ce sanitizer
    garantit que tout flottant non-fini est remplacé par None (→ null en JSON).
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


try:
    import websockets                    # pip install websockets>=11
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    websockets = None  # type: ignore


class DashboardServer:
    """Serveur WebSocket dans un thread daemon.

    push(state) : non-bloquant, thread-safe, appelable à 20 Hz depuis le main thread.
    start()     : démarre le thread + l'event loop asyncio.
    stop()      : signal d'arrêt (le thread daemon s'éteint dans ≤ 50ms).
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host = host
        self._port = port
        # SimpleQueue : unbounded, thread-safe, put_nowait ne lève jamais Full
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Route globale : sérialisée 1× par set_route(), envoyée à chaque nouveau client.
        # Affectation atomique sous CPython (GIL) → thread-safe sans verrou.
        self._route_msg: Optional[str] = None

    # ------------------------------------------------------------------
    # API publique — appelée depuis la boucle CARLA (JAMAIS bloquant)
    # ------------------------------------------------------------------

    def push(self, state: dict) -> None:
        """Enqueue un état. Fire-and-forget. Jamais bloquant."""
        self._queue.put_nowait(state)

    def set_route(self, points: list) -> None:
        """Stocke la route globale à envoyer à chaque nouveau client.

        Appelée UNE SEULE FOIS depuis le thread CARLA (avant la boucle principale),
        après que GlobalPlannerAgent.compute_route() a rempli blackboard.route.route.
        Thread-safe : l'affectation de self._route_msg est atomique sous CPython (GIL).

        Format attendu : [[x0, y0], [x1, y1], ...] en coordonnées CARLA (mètres).
        Le client reçoit {type: "route", points: [[x,y], ...]} à la connexion.
        """
        clean = [
            [round(float(p[0]), 1), round(float(p[1]), 1)]
            for p in points
        ]
        self._route_msg = json.dumps({"type": "route", "points": clean})
        print(
            f"[DASHBOARD] Route stockee : {len(clean)} pts  "
            f"{len(self._route_msg)} bytes"
        )

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Démarre le thread WebSocket. Retourne False si 'websockets' absent."""
        if not _WS_AVAILABLE:
            logger.warning(
                "[DASHBOARD] Package 'websockets' introuvable — dashboard désactivé. "
                "Installez avec : pip install websockets>=11"
            )
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DashboardWS"
        )
        self._thread.start()
        logger.info("[DASHBOARD] Thread demarre -> ws://%s:%d", self._host, self._port)
        return True

    def stop(self) -> None:
        """Signal d'arrêt propre.

        Positionne _running=False. Le broadcaster sort de sa boucle while à la
        prochaine itération (≤50ms), _serve() retourne normalement, le contexte
        websockets.serve se ferme proprement. Aucune erreur asyncio.
        Pas besoin de join() : le thread est daemon.
        """
        self._running = False

    # ------------------------------------------------------------------
    # Interne — tourne dans le thread daemon
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:
            # BaseException (pas seulement Exception) pour attraper CancelledError
            # en Python 3.8+ (où CancelledError n'est plus sous-classe d'Exception)
            import traceback
            print(f"[DASHBOARD] CRASH serveur: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            print("[DASHBOARD] Event loop ferme — thread WS termine")
            self._loop.close()

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            # ping_interval=None : désactive le keepalive WS applicatif.
            # Sur localhost, le TCP OS keepalive suffit. Evite les faux timeouts
            # quand l'event loop asyncio est intermittent (Python 3.7 / Windows).
            ping_interval=None,
            # open_timeout=None : desactive le timeout du handshake d'ouverture
            # (defaut=10s). Sous GIL starvation CARLA (71 Hz), le thread asyncio
            # peut mettre >10s a traiter le HTTP Upgrade lors d'une reconnexion.
            # None = attendre indefiniment → la connexion finit toujours par s'etablir.
            open_timeout=None,
        ):
            print(f"[DASHBOARD] Serveur WS sur ws://{self._host}:{self._port}")
            await self._broadcaster()
        # Si on arrive ici, websockets.serve a ferme (apres _broadcaster() sorti)
        print("[DASHBOARD] _serve() termine — contexte websockets.serve ferme")

    async def _handler(self, websocket) -> None:
        """Gère une connexion client jusqu'à déconnexion.

        wait_closed() retourne quand le TCP socket est fermé (proprement ou non).
        Le websocket est retiré de _clients dans finally → le broadcaster ne lui
        enverra plus rien. La simulation n'est pas affectée.
        """
        self._clients.add(websocket)
        n = len(self._clients)
        print(f"[DASHBOARD] Client connecté ({n} total)")
        # Envoie la route globale immédiatement — avant le premier tick broadcast.
        # Permet à la mini-map de s'initialiser dès la connexion, même si peu de
        # ticks ont été envoyés (ou si le client se reconnecte en cours de route).
        if self._route_msg is not None:
            try:
                await websocket.send(self._route_msg)
            except Exception as exc:
                print(
                    f"[DASHBOARD] envoi route echec: "
                    f"{type(exc).__name__}: {exc}"
                )
        try:
            await websocket.wait_closed()
        except Exception as exc:
            # CancelledError (Python 3.7 : sous-classe de Exception) ou autre erreur asyncio
            print(f"[DASHBOARD] wait_closed() exception: {type(exc).__name__}: {exc}")
        finally:
            self._clients.discard(websocket)
            close_code   = getattr(websocket, "close_code",   "?")
            close_reason = getattr(websocket, "close_reason", "")
            print(
                f"[DASHBOARD] Client déconnecté "
                f"(code={close_code}  reason={close_reason!r}  "
                f"{len(self._clients)} restant(s))"
            )

    async def _broadcaster(self) -> None:
        """Draine la queue et envoie le DERNIER état à tous les clients.

        Pattern latest-value : tous les états intermédiaires accumulés depuis le
        dernier broadcast sont droppés — seul le plus récent est envoyé.
        """
        import traceback as _tb
        print("[DASHBOARD] _broadcaster() demarre")
        try:
            while self._running:
                # Drain complet : on ne garde que le dernier
                latest = None
                while True:
                    try:
                        latest = self._queue.get_nowait()
                    except queue.Empty:
                        break

                if latest is not None and self._clients:
                    msg = None
                    try:
                        msg = json.dumps(_sanitize(latest), allow_nan=False)
                    except Exception as enc_exc:
                        print(f"[DASHBOARD] serialisation erreur: {type(enc_exc).__name__}: {enc_exc}")

                    if msg is not None:
                        dead: Set = set()
                        for ws in list(self._clients):
                            try:
                                await ws.send(msg)
                            except Exception as send_exc:
                                print(
                                    f"[DASHBOARD] send() echec: "
                                    f"{type(send_exc).__name__}: {send_exc}"
                                )
                                dead.add(ws)
                        self._clients -= dead

                await asyncio.sleep(0.05)   # broadcast max 20 Hz

        except BaseException as exc:
            # Attrape CancelledError (BaseException en Python 3.8+) ET toute autre
            # exception qui aurait echappe aux try/except internes. Sans ce bloc,
            # une exception ici ferait sortir _broadcaster() silencieusement, ce qui
            # fermerait websockets.serve et enverrait 1001 a tous les clients.
            print(f"[DASHBOARD] _broadcaster() EXCEPTION INATTENDUE: {type(exc).__name__}: {exc}")
            _tb.print_exc()
            raise  # re-propage pour que _run() le logue aussi

        finally:
            print(f"[DASHBOARD] _broadcaster() termine (running={self._running})")
