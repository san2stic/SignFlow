interface VideoPlayerProps {
  src: string;
}

export function VideoPlayer({ src }: VideoPlayerProps): JSX.Element {
  return <video src={src} controls className="w-full rounded-card" />;
}
