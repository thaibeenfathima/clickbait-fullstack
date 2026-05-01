import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import Card from '../components/Card';
import AnimatedBackground from '../components/AnimatedBackground';

export default function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      // Use sample data (no backend analytics endpoint)
      setAnalytics(sampleAnalytics);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Sample data for demonstration
  const sampleAnalytics = {
    total_headlines: 1247,
    clickbait_count: 456,
    non_clickbait_count: 791,
    sentiment_distribution: [
      { name: 'Positive', value: 467, color: '#10b981' },
      { name: 'Negative', value: 389, color: '#ef4444' },
      { name: 'Neutral', value: 391, color: '#6b7280' },
    ],
    daily_analysis: [
      { day: 'Mon', count: 145 },
      { day: 'Tue', count: 168 },
      { day: 'Wed', count: 152 },
      { day: 'Thu', count: 178 },
      { day: 'Fri', count: 192 },
      { day: 'Sat', count: 138 },
      { day: 'Sun', count: 129 },
    ],
  };

  const data = analytics || sampleAnalytics;
  const clickbaitPercentage = ((data.clickbait_count / data.total_headlines) * 100).toFixed(1);

  if (loading) {
    return (
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      </main>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative">
      <AnimatedBackground />
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Analytics Dashboard</h1>
        <p className="text-lg text-gray-300">Overview of all analyzed headlines and predictions</p>
      </div>

      {/* KPI Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Total Headlines */}
        <Card className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-gray-300 mb-1">Total Headlines Analyzed</p>
              <p className="text-3xl font-bold text-blue-400">{data.total_headlines.toLocaleString()}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-400 opacity-50" />
          </div>
        </Card>

        {/* Clickbait Count */}
        <Card className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-gray-300 mb-1">Clickbait Headlines</p>
              <p className="text-3xl font-bold text-red-400">{data.clickbait_count.toLocaleString()}</p>
              <p className="text-xs text-gray-400 mt-2">{clickbaitPercentage}% of total</p>
            </div>
            <AlertCircle className="w-8 h-8 text-red-500 opacity-50" />
          </div>
        </Card>

        {/* Non-Clickbait Count */}
        <Card className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-gray-300 mb-1">Non-Clickbait Headlines</p>
              <p className="text-3xl font-bold text-green-400">{data.non_clickbait_count.toLocaleString()}</p>
              <p className="text-xs text-gray-400 mt-2">{(100 - clickbaitPercentage).toFixed(1)}% of total</p>
            </div>
            <TrendingDown className="w-8 h-8 text-green-500 opacity-50" />
          </div>
        </Card>

        {/* Accuracy */}
        <Card className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-gray-300 mb-1">Detection Accuracy</p>
              <p className="text-3xl font-bold text-purple-400">98%</p>
              <p className="text-xs text-gray-400 mt-2">Model Performance</p>
            </div>
            <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">
              ✓
            </div>
          </div>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-8 mb-8">
        {/* Daily Analysis Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-white mb-6">Headlines Analyzed (Weekly)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.daily_analysis}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="day" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #374151', borderRadius: '8px', color: '#f3f4f6' }}
              />
              <Bar dataKey="count" fill="#38bdf8" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Sentiment Distribution */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-white mb-6">Sentiment Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data.sentiment_distribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.sentiment_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Clickbait vs Non-Clickbait */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-6">Clickbait vs Non-Clickbait</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={[
              {
                name: 'Classification',
                clickbait: data.clickbait_count,
                non_clickbait: data.non_clickbait_count,
              },
            ]}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #374151', borderRadius: '8px', color: '#f3f4f6' }}
            />
            <Legend />
            <Bar dataKey="clickbait" fill="#ef4444" name="Clickbait" radius={[8, 8, 0, 0]} />
            <Bar dataKey="non_clickbait" fill="#10b981" name="Non-Clickbait" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Recent Activity */}
      <Card className="p-6 mt-8">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
        <div className="space-y-3 text-sm text-gray-300">
          <p>✓ 150 headlines analyzed in the last hour</p>
          <p>✓ 48 clickbait headlines detected</p>
          <p>✓ 89 positive sentiment articles</p>
          <p>✓ System uptime: 99.9%</p>
        </div>
      </Card>
    </main>
  );
}
