import { Suspense, lazy } from "react";
import { Navigate, createBrowserRouter, isRouteErrorResponse, useRouteError } from "react-router-dom";

import MainLayout from "./components/layout/MainLayout";
import ProtectedRoute from "./components/ProtectedRoute";
import Login from "./pages/Login";
import Register from "./pages/Register";

const DashboardPage = lazy(async () => {
  const module = await import("./pages/DashboardPage");
  return { default: module.DashboardPage };
});
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
const TrainingSessionPage = lazy(async () => {
  const module = await import("./pages/TrainingSessionPage");
  return { default: module.TrainingSessionPage };
});
const SettingsPage = lazy(async () => {
  const module = await import("./pages/SettingsPage");
  return { default: module.SettingsPage };
});
const Profile = lazy(() => import("./pages/Profile"));

const StudioPage = lazy(async () => {
  const module = await import("./pages/StudioPage");
  return { default: module.StudioPage };
});
const StudioSessionPage = lazy(async () => {
  const module = await import("./pages/StudioSessionPage");
  return { default: module.StudioSessionPage };
});
const VideoAnnotationPage = lazy(async () => {
  const module = await import("./pages/VideoAnnotationPage");
  return { default: module.VideoAnnotationPage };
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
      { path: "dashboard", element: withSuspense(<DashboardPage />) },
      { path: "translate", element: withSuspense(<TranslatePage />) },
      { path: "dictionary", element: withSuspense(<DictionaryPage />) },
      { path: "training", element: withSuspense(<TrainPage />) },
      { path: "training/:sessionId", element: withSuspense(<TrainingSessionPage />) },
      { path: "train", element: withSuspense(<TrainPage />) },
      { path: "settings", element: withSuspense(<SettingsPage />) },
      { path: "profile", element: withSuspense(<Profile />) },
      // Studio routes
      { path: "studio", element: withSuspense(<StudioPage />) },
      { path: "studio/sessions/:sessionId", element: withSuspense(<StudioSessionPage />) },
      {
        path: "studio/sessions/:sessionId/videos/:videoId/annotate",
        element: withSuspense(<VideoAnnotationPage />),
      },
      { path: "*", element: <Navigate to="/dashboard" replace /> },
    ],
  },
]);
