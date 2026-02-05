import { FocusRail, type FocusRailItem } from "@/components/ui/focus-rail"

const DEMO_ITEMS: FocusRailItem[] = [
  {
    id: 1,
    title: "Detect it",
    description: "— automatically",
    meta: "No 1.",
    imageSrc: "/image1.png",
    href: "/camera",
  },
  {
    id: 2,
    title: "Restyle it with AI",
    description: "— intelligently",
    meta: "No 2.",
    imageSrc: "/image2.png",
    href: "/camera",
  },
  {
    id: 3,
    title: "Turn it into a ready-to-use 3D asset",
    description: "— instantly",
    meta: "No 3.",
    imageSrc: "/image3.png",
    href: "/camera",
  },
]

const FocusRailDemo = () => {
  return (
    <main className="min-h-screen overflow-x-hidden w-full bg-neutral-950 flex flex-col items-center justify-center py-20">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">Ready to Change Your World?</h1>
        <p className="text-neutral-400">Point your camera at anything.</p>
      </div>

      <FocusRail items={DEMO_ITEMS} autoPlay={false} loop={true} />
    </main>
  )
}

export default FocusRailDemo
