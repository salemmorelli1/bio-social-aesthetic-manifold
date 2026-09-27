import { ArrowUpRight, ScanFace } from "lucide-react";
import FaceCapture from "@/components/dashboard/FaceCapture";

export default function Home() {
  return (
    <main className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="AME capture home">
          <span className="brand-symbol"><ScanFace size={22} strokeWidth={1.7} aria-hidden="true" /></span>
          <span>AME <small>Research preview</small></span>
        </a>
        <span className="header-note">Aesthetic-Morphometrics-Engine · Measurement preview</span>
      </header>

      <div className="content-grid">
        <section className="intro" aria-labelledby="intro-title">
          <p className="eyebrow">An instrument for observation</p>
          <h1 id="intro-title">Facial geometry, <em>without the guesswork.</em></h1>
          <p className="lead">With your choice of photo or camera, this preview maps one face and reports projected geometry entirely in your browser. It makes no claim about beauty, identity, ancestry, sex, health, or what someone should change.</p>
          <div className="step-rail" aria-label="Project stages">
            <span className="step-current">01 · Foundation</span>
            <span className="step-current">02 · Capture</span>
            <span className="step-current">03 · Projected measurements</span>
          </div>
          <div className="scope-note">
            <span className="scope-line" />
            <p>Perspective, camera distance, expression, occlusion, and landmark error can change apparent proportions. The amber three-point line is an image-plane guide—not an anatomical midline or pose estimate.</p>
          </div>
          <a className="legacy-link" href="https://salemmorelli1.github.io/bio-social-aesthetic-manifold/" target="_blank" rel="noopener noreferrer">
            Explore the existing descriptive shape laboratory <ArrowUpRight size={17} aria-hidden="true" />
          </a>
        </section>

        <FaceCapture />
      </div>
      <footer className="site-footer">Open-source research preview · Local by design · No automated aesthetic judgment</footer>
    </main>
  );
}
