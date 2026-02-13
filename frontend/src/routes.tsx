import { Suspense, lazy } from "react";
import { Navigate, createBrowserRouter, isRouteErrorResponse, useRouteError } from "react-router-dom";

import { PageShell } from "./components/layout/PageShell";

const TranslatePage = lazy(async () => {
  const module = await import("./pages/TranslatePage");
  return { default: module.TranslatePage };
});

const TrainPage = lazy(async () => {
  const module = await import("./pages/TrainPage");
  return { default: module.TrainPage };
});

const DictionaryPage = lazy(async () => {
  const module = await import("./pages/DictionaryPage");
  return { default: module.DictionaryPage };
});

const DashboardPage = lazy(async () => {
  const module = await import("./pages/DashboardPage");
  return { default: module.DashboardPage };
});

const SettingsPage = lazy(async () => {
  const module = await import("./pages/SettingsPage");
  return { default: module.SettingsPage };
});

function RouteFallback(): JSX.Element {
  return (
    <div className="flex min-h-[35vh] items-center justify-center px-4 text-center text-sm text-slate-400">
      Loading...
    </div>
  );
}

function withSuspense(element: JSX.Element): JSX.Element {
  return <Suspense fallback={<RouteFallback />}>{element}</Suspense>;
}

function RouteErrorBoundary(): JSX.Element {
  const error = useRouteError();
  let message = "An unexpected error occurred.";

  if (isRouteErrorResponse(error)) {
    message = `${error.status} ${error.statusText}`;
  } else if (error instanceof Error) {
    message = error.message;
  }

  return (
    <div className="mx-auto flex min-h-[55vh] max-w-xl flex-col items-center justify-center gap-4 px-4 text-center">
      <h2 className="font-heading text-2xl text-red-300">Application Error</h2>
      <p className="text-sm text-slate-300">{message}</p>
      <button className="touch-btn bg-primary text-white" onClick={() => window.location.assign("/translate")}>
        Go to Translate
      </button>
    </div>
  );
}

export const routeItems = [
  { path: "/translate", label: "Translate", icon: "📹" },
  { path: "/train", label: "Train", icon: "🎯" },
  { path: "/dictionary", label: "Dictionary", icon: "📖" },
  { path: "/dashboard", label: "Dashboard", icon: "📊" },
  { path: "/settings", label: "Settings", icon: "⚙️" }
];

export const router = createBrowserRouter([
  {
    path: "/",
    element: <PageShell />,
    errorElement: <RouteErrorBoundary />,
    children: [
      { index: true, element: <Navigate to="/translate" replace /> },
      { path: "translate", element: withSuspense(<TranslatePage />) },
      { path: "train", element: withSuspense(<TrainPage />) },
      { path: "dictionary", element: withSuspense(<DictionaryPage />) },
      { path: "dashboard", element: withSuspense(<DashboardPage />) },
      { path: "settings", element: withSuspense(<SettingsPage />) },
      { path: "*", element: <Navigate to="/translate" replace /> }
    ]
  }
]);
