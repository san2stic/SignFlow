import { GraduationCap } from "lucide-react";

export default function Training() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Training & Labeling</h1>
        <p className="text-slate-600">Train models and label sign language data</p>
      </div>
      <div className="bg-white rounded-2xl p-12 border border-gray-200 text-center">
        <div className="w-16 h-16 bg-gradient-to-br from-amber-100 to-orange-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <GraduationCap className="w-8 h-8 text-amber-600" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">Training Tools Coming Soon</h3>
        <p className="text-slate-600">Training and labeling features are under development.</p>
      </div>
    </div>
  );
}
