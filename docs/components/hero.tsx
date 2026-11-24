'use client';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import type { Hero as HeroHandle, HeroColor, HeroItem, HeroLayout, MountHero } from '@/hero/types';
import { appDescription, appName, docsRoute, examplesRoute, githubUrl } from '@/lib/shared';

// Any CSS color as RGBA: the browser paints it and the pixel is read back,
// whatever the notation the computed style uses.
const makeColorReader = () => {
  const context = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true })!;
  return (css: string): HeroColor => {
    context.clearRect(0, 0, 1, 1);
    context.fillStyle = css;
    context.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data;
    // The pixel is not premultiplied, but a transparent one has no color left.
    return a === 0 ? [0, 0, 0, 0] : [r / 255, g / 255, b / 255, a / 255];
  };
};

/**
 * The home page. Its title, text and buttons are real elements: they are the
 * layout, the links and what a screen reader finds. Once the hero is running
 * they are made transparent, and Skia draws them again in two textures: what
 * is marked data-hero-front in one, everything else in the other. A WebGPU
 * pass draws a ribbon between the two, which rises over both once it loops:
 * see hero/hero.ts.
 *
 * Without WebGPU, or without the wasm module, the elements simply stay.
 */
export function Hero() {
  const stage = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    const root = stage.current;
    const target = canvas.current;
    if (!root || !target) return;
    let hero: HeroHandle | null = null;
    let disposed = false;
    const readColor = makeColorReader();
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');

    // What the elements marked data-hero look like right now.
    const measure = (): HeroLayout => {
      const frame = root.getBoundingClientRect();
      const items = Array.from(root.querySelectorAll<HTMLElement>('[data-hero]')).map(
        (element): HeroItem => {
          const box = element.getBoundingClientRect();
          const style = getComputedStyle(element);
          return {
            text: element.textContent ?? '',
            x: box.left - frame.left,
            y: box.top - frame.top,
            width: box.width,
            height: box.height,
            fontSize: parseFloat(style.fontSize),
            fontWeight: Number(style.fontWeight) || 400,
            color: readColor(style.color),
            background: readColor(style.backgroundColor),
            border: readColor(style.borderTopColor),
            borderWidth: parseFloat(style.borderTopWidth),
            radius: parseFloat(style.borderTopLeftRadius),
            // data-hero-front="": all of it. data-hero-front="word": that word.
            onTop: element.dataset.heroFront === '',
            front: element.dataset.heroFront ?? '',
          };
        },
      );
      return {
        width: frame.width,
        height: frame.height,
        pixelRatio: Math.min(window.devicePixelRatio, 2),
        dark: document.documentElement.classList.contains('dark'),
        still: motion.matches,
        items,
      };
    };
    const update = () => hero?.update(measure());

    const resize = new ResizeObserver(update);
    const theme = new MutationObserver(update);
    (async () => {
      // Built apart from the site, with the examples: scripts/examples.mjs.
      const url = `${examplesRoute}/hero.js`;
      const { mountHero } = (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ url)) as {
        mountHero: MountHero;
      };
      // The web font changes the layout when it arrives.
      await document.fonts.ready;
      // React runs an effect twice in development: the first run is already
      // cleaned up by now, and must not touch the canvas.
      if (disposed) return;
      const mounted = await mountHero(target, examplesRoute, measure());
      if (disposed) {
        mounted.destroy();
        return;
      }
      hero = mounted;
      setRunning(true);
      resize.observe(root);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
      motion.addEventListener('change', update);
    })().catch((error: unknown) => {
      // The page works without its hero.
      console.info('The hero is not running:', error);
    });

    return () => {
      disposed = true;
      resize.disconnect();
      theme.disconnect();
      motion.removeEventListener('change', update);
      hero?.destroy();
    };
  }, []);

  // Once Skia draws an element it is hidden, but it stays in place: it still
  // takes clicks, and it shows again while it has the keyboard focus.
  const drawn = running ? 'opacity-0 focus-visible:opacity-100' : '';

  return (
    <main
      ref={stage}
      className="relative flex flex-1 flex-col items-center justify-center gap-6 overflow-hidden px-6 py-16 text-center"
    >
      <canvas
        ref={canvas}
        aria-hidden="true"
        className={`pointer-events-none absolute inset-0 h-full w-full ${running ? '' : 'invisible'}`}
      />
      <p
        data-hero
        className={`rounded-full border px-3 py-1 text-xs font-medium text-fd-muted-foreground ${drawn}`}
      >
        Experimental: no stable API yet
      </p>
      <h1 data-hero data-hero-front="skia" className={`text-6xl font-bold tracking-tight sm:text-8xl ${drawn}`}>
        {appName}
      </h1>
      <p
        data-hero
        data-hero-front=""
        className={`max-w-xl text-lg text-fd-muted-foreground ${drawn}`}
      >
        {appDescription} Zero-copy interop with 3D scenes, compute shaders, and video and camera
        pipelines.
      </p>
      <div className="flex flex-row flex-wrap justify-center gap-3">
        <Link
          data-hero
          href={docsRoute}
          className={`rounded-full bg-fd-primary px-5 py-2 text-sm font-medium text-fd-primary-foreground ${drawn}`}
        >
          Read the docs
        </Link>
        <a
          data-hero
          href={githubUrl}
          rel="noreferrer noopener"
          target="_blank"
          className={`rounded-full border bg-fd-background px-5 py-2 text-sm font-medium ${drawn}`}
        >
          GitHub
        </a>
      </div>
    </main>
  );
}
