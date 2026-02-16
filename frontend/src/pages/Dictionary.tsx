import { BookOpen } from "lucide-react";

export default function Dictionary() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Dictionary</h1>
        <p className="text-slate-600">Browse and search sign language dictionary</p>
      </div>
      <div className="bg-white rounded-2xl p-12 border border-gray-200 text-center">
        <div className="w-16 h-16 bg-gradient-to-br from-teal-100 to-cyan-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <BookOpen className="w-8 h-8 text-teal-600" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">Dictionary Coming Soon</h3>
        <p className="text-slate-600">Sign dictionary features are under development.</p>
      </div>
    </div>
  );
}
