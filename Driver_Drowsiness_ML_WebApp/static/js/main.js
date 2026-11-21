/**
 * DrowsyGuard - main.js
 * Global JS: navbar scroll effect, intersection observer for animations
 */

// ── Navbar scroll shadow ──
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
      navbar.style.boxShadow = '0 4px 32px rgba(0,0,0,0.5)';
    } else {
      navbar.style.boxShadow = 'none';
    }
  }, { passive: true });
}

// ── Intersection Observer: trigger animations when in view ──
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.animate-fade-up, .animate-fade-left').forEach(el => {
  el.style.animationPlayState = 'paused';
  observer.observe(el);
});

// ── Metric bar fill animation on scroll ──
const metricObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const fill = entry.target.querySelector('.metric-fill');
      if (fill) {
        const target = fill.style.width;
        fill.style.width = '0';
        requestAnimationFrame(() => {
          setTimeout(() => { fill.style.width = target; }, 50);
        });
      }
      metricObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.metric-card').forEach(el => metricObserver.observe(el));

// ── Ring arc animation ──
const ringArc = document.querySelector('.ring-arc');
if (ringArc) {
  const r = 85;
  const circ = 2 * Math.PI * r;        // ≈ 534
  const target = 0.9883;               // 98.83% accuracy
  const offset = circ * (1 - target);  // ≈ 6.4

  // Start fully "empty", animate to filled
  ringArc.style.strokeDashoffset = circ;
  setTimeout(() => {
    ringArc.style.transition = 'stroke-dashoffset 2s cubic-bezier(0.4,0,0.2,1)';
    ringArc.style.strokeDashoffset = offset;
  }, 300);
}
