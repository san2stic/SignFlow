import { useState } from "react";
import { useAuthStore } from "../stores/authStore";
import { Languages, BookOpen, Target, TrendingUp, CheckCircle, Clock, FileText, X } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const weeklyData = [
  { day: "Sun", count: 12 },
  { day: "Mon", count: 35 },
  { day: "Tue", count: 28 },
  { day: "Wed", count: 42 },
  { day: "Thu", count: 38 },
  { day: "Fri", count: 51 },
  { day: "Sat", count: 45 },
];

const recentActivity = [
  {
    id: 1,
    type: "translated",
    text: 'Translated "Bonjour"',
    time: "2 mins ago",
    icon: CheckCircle,
    iconColor: "text-green-600",
    bgColor: "bg-green-100",
  },
  {
    id: 2,
    type: "labeled",
    text: 'Labeled 5 new clips for "MERCI"',
    time: "15 mins ago",
    icon: Clock,
    iconColor: "text-amber-600",
    bgColor: "bg-amber-100",
  },
  {
    id: 3,
    type: "dictionary",
    text: 'Dictionary updated: "TRAVAIL"',
    time: "1 hour ago",
    icon: FileText,
    iconColor: "text-slate-600",
    bgColor: "bg-slate-100",
  },
];

const smartSuggestions = [
  { label: "FINE", image: "🤙" },
  { label: "GOOD", image: "👍" },
  { label: "BAD", image: "👎" },
];

export default function Dashboard() {
  const user = useAuthStore((state) => state.user);
  const [showTranslatePanel, setShowTranslatePanel] = useState(true);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Main Dashboard Area */}
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Dashboard</h1>
            <p className="text-slate-600">
              Welcome back, <span className="font-semibold">{user?.username || "User"}</span>!
            </p>
            <p className="text-sm text-slate-500">Here's your SignFlow overview.</p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {/* Translations Today */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600 mb-1">Translations Today</p>
                  <p className="text-3xl font-bold text-slate-900">145</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-xl flex items-center justify-center">
                  <Languages className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span className="text-sm text-green-600 font-medium">+12%</span>
              </div>
            </div>

            {/* Pending Labels */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600 mb-1">Pending Labels</p>
                  <p className="text-3xl font-bold text-slate-900">32</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl flex items-center justify-center">
                  <Clock className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>

            {/* Dictionary Entries */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600 mb-1">Dictionary Entries</p>
                  <p className="text-3xl font-bold text-slate-900">1,250</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-teal-400 to-cyan-500 rounded-xl flex items-center justify-center">
                  <BookOpen className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>

            {/* Model Accuracy */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600 mb-1">Model Accuracy</p>
                  <p className="text-3xl font-bold text-slate-900">94.5%</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-400 to-purple-500 rounded-xl flex items-center justify-center">
                  <Target className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Recent Activity */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
              <h3 className="text-lg font-bold text-slate-900 mb-4">Recent Activity</h3>
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-start gap-4">
                    <div className={`w-10 h-10 ${activity.bgColor} rounded-lg flex items-center justify-center flex-shrink-0`}>
                      <activity.icon className={`w-5 h-5 ${activity.iconColor}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-900">{activity.text}</p>
                      <p className="text-xs text-slate-500 mt-1">{activity.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Weekly Translation Volume */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200">
              <h3 className="text-lg font-bold text-slate-900 mb-4">Weekly Translation Volume</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={weeklyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="day" stroke="#94a3b8" style={{ fontSize: "12px" }} />
                  <YAxis stroke="#94a3b8" style={{ fontSize: "12px" }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: "#1e293b",
                      border: "none",
                      borderRadius: "8px",
                      color: "#fff"
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#14b8a6" 
                    strokeWidth={3}
                    dot={{ fill: "#14b8a6", r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Live Translation Assist Panel */}
      {showTranslatePanel && (
        <aside className="w-96 bg-white border-l border-gray-200 flex flex-col h-screen overflow-hidden">
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-bold text-slate-900">Live Translation Assist</h2>
            <button
              onClick={() => setShowTranslatePanel(false)}
              className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-5 h-5 text-slate-600" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Video Preview */}
            <div className="aspect-video bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl overflow-hidden relative">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-48 h-48 border-4 border-green-500 rounded-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-2">
                      <div className="w-12 h-12 bg-green-500 rounded-full"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="absolute top-4 right-4 px-3 py-1 bg-green-500 rounded-full flex items-center gap-2">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="text-xs font-semibold text-white">LIVE</span>
              </div>
            </div>

            {/* Confidence Score */}
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-700">Confidence score</span>
                <span className="text-lg font-bold text-green-600">98%</span>
              </div>
              <p className="text-2xl font-bold text-slate-900 uppercase tracking-wide">
                HELLO, HOW ARE YOU TODAY?
              </p>
            </div>

            {/* Smart Suggestions */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-3">Smart Suggestions</h3>
              <div className="grid grid-cols-3 gap-3">
                {smartSuggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    className="aspect-square bg-slate-100 hover:bg-slate-200 rounded-xl flex flex-col items-center justify-center transition-colors border border-slate-200"
                  >
                    <span className="text-4xl mb-2">{suggestion.image}</span>
                    <span className="text-xs font-semibold text-slate-700">{suggestion.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Action Button */}
            <button className="w-full bg-gradient-to-r from-slate-700 to-slate-900 hover:from-slate-800 hover:to-slate-950 text-white font-semibold py-3 rounded-xl transition-all shadow-lg">
              Correct & Label Clip
            </button>
          </div>
        </aside>
      )}

      {/* Floating Button to Reopen Panel */}
      {!showTranslatePanel && (
        <button
          onClick={() => setShowTranslatePanel(true)}
          className="fixed bottom-8 right-8 w-14 h-14 bg-gradient-to-br from-teal-500 to-cyan-600 text-white rounded-full shadow-2xl hover:scale-110 transition-transform flex items-center justify-center"
        >
          <Languages className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}
