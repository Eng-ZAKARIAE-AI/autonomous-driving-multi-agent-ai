import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface RewardChartProps {
  data: { step: number; reward: number }[];
}

export const RewardChart = ({ data }: RewardChartProps) => {
  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1f6feb" stopOpacity={0.25}/>
              <stop offset="100%" stopColor="#1f6feb" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#21262d" vertical={false} />
          <XAxis 
            dataKey="step" 
            hide 
          />
          <YAxis 
            stroke="#484f58" 
            fontSize={9} 
            tickLine={false} 
            axisLine={false}
            tick={{ fontFamily: 'IBM Plex Mono' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #21262d', borderRadius: '6px', fontSize: '11px', fontFamily: 'IBM Plex Sans' }}
            itemStyle={{ color: '#58a6ff' }}
            labelStyle={{ display: 'none' }}
          />
          <Area 
            type="monotone" 
            dataKey="reward" 
            stroke="#58a6ff" 
            strokeWidth={2} 
            fillOpacity={1} 
            fill="url(#areaGrad)" 
            isAnimationActive={true}
            dot={{ r: 3, fill: '#58a6ff', strokeWidth: 0 }}
            activeDot={{ r: 5, fill: '#58a6ff', strokeWidth: 2, stroke: '#0d1117' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
