import { promises as fs } from "node:fs";
import path from "node:path";

import sharp from "sharp";

const FALLBACK_BLUR =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAKCAQAAAB+fV9XAAAAL0lEQVR42mP8z8AARAwMjMxMDAxwA4MDA4P5nxkYGP5nZGRk+J8ZGBl+QwAAiOsI5JgkKX8AAAAASUVORK5CYII=";

export async function getBlurDataURL(
  relativePublicPath: string
): Promise<string> {
  try {
    const imagePath = path.join(process.cwd(), "public", relativePublicPath);
    await fs.access(imagePath);

    const tiny = await sharp(imagePath)
      .resize(20)
      .blur(8)
      .jpeg({ quality: 40 })
      .toBuffer();

    return `data:image/jpeg;base64,${tiny.toString("base64")}`;
  } catch {
    return FALLBACK_BLUR;
  }
}
