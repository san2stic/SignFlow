import { Settings as SettingsIcon } from "lucide-react";

export default function Settings() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Settings</h1>
        <p className="text-slate-600">Manage your application preferences</p>
      </div>
      <div className="bg-white rounded-2xl p-12 border border-gray-200 text-center">
        <div className="w-16 h-16 bg-gradient-to-br from-slate-100 to-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <SettingsIcon className="w-8 h-8 text-slate-600" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">Settings Coming Soon</h3>
        <p className="text-slate-600">Settings panel is under development.</p>
      </div>
    </div>
  );
}
