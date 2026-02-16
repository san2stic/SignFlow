import { Suspense, lazy } from "react";
import { Navigate, createBrowserRouter, isRouteErrorResponse, useRouteError } from "react-router-dom";

import MainLayout from "./components/layout/MainLayout";
import ProtectedRoute from "./components/ProtectedRoute";
import Login from "./pages/Login";
import Register from "./pages/Register";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const TranslatePage = lazy(async () => {
  const module = await import("./pages/TranslatePage");
  return { default: module.TranslatePage };
});
const DictionaryPage = lazy(async () => {
  const module = await import("./pages/DictionaryPage");
  return { default: module.DictionaryPage };
});
const TrainPage = lazy(async () => {
  const module = await import("./pages/TrainPage");
  return { default: module.TrainPage };
});
const SettingsPage = lazy(async () => {
  const module = await import("./pages/SettingsPage");
  return { default: module.SettingsPage };
});
const Profile = lazy(() => import("./pages/Profile"));

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

export const router = createBrowserRouter([
  {
    path: "/login",
    element: <Login />,
  },
  {
    path: "/register",
    element: <Register />,
  },
  {
    path: "/",
    element: (
      <ProtectedRoute>
        <MainLayout />
      </ProtectedRoute>
    ),
    errorElement: <RouteErrorBoundary />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: "dashboard", element: withSuspense(<Dashboard />) },
      { path: "translate", element: withSuspense(<TranslatePage />) },
      { path: "dictionary", element: withSuspense(<DictionaryPage />) },
      { path: "training", element: withSuspense(<TrainPage />) },
      { path: "train", element: withSuspense(<TrainPage />) },
      { path: "settings", element: withSuspense(<SettingsPage />) },
      { path: "profile", element: withSuspense(<Profile />) },
      { path: "*", element: <Navigate to="/dashboard" replace /> },
    ],
  },
]);
