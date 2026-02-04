import { useEffect } from "react";

export function Helmet({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  useEffect(() => {
    document.title = title;

    let meta = document.querySelector('meta[name="description"]') as HTMLMetaElement | null;
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "description";
      document.head.appendChild(meta);
    }
    meta.content = description;
  }, [title, description]);

  return null;
}
