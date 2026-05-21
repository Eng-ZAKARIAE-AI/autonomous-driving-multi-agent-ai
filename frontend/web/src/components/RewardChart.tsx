import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface RewardChartProps {
  data: { step: number; reward: number }[];
}

export const RewardChart = ({ data }: RewardChartProps) => {
  return (
    <div className="bg-surface border border-slate-800 rounded-xl p-6 h-64 shadow-sm">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">Step Reward</h3>
      <div className="w-full h-full pb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis 
              dataKey="step" 
              hide 
            />
            <YAxis 
              stroke="#94a3b8" 
              fontSize={12} 
              tickLine={false} 
              axisLine={false}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              itemStyle={{ color: '#6366f1' }}
              labelStyle={{ display: 'none' }}
            />
            <Line 
              type="monotone" 
              dataKey="reward" 
              stroke="#6366f1" 
              strokeWidth={2} 
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
