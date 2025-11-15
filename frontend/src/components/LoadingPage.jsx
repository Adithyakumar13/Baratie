import React, { useState, useEffect } from 'react';

export default function LoadingPage({ onLoadingComplete }) {
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setFadeOut(true);
      setTimeout(() => {
        onLoadingComplete();
      }, 1000); // Duration of swipe right animation
    }, 5000);

    return () => clearTimeout(timer);
  }, [onLoadingComplete]);

  return (
    <div
      className={`relative w-full h-screen overflow-hidden flex items-center justify-center transition-all duration-1000 ${
        fadeOut ? 'translate-x-full opacity-0' : 'translate-x-0 opacity-100'
      }`}
    >
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage: `url(/Ramen.png)`,
          opacity: 0.8,
        }}
      />

      {/* Title */}
      <div className="relative z-10 text-center">
        <h1
          style={{
            color: 'white',
            fontFamily: 'Nunito, sans-serif',
            fontSize: window.innerWidth < 768 ? '58px' : '130px',
            fontWeight: 'bold',
            WebkitTextStroke: '5px #001B48',
            textStroke: '5px #001B48',
            letterSpacing: '0.02em',
            margin: 0,
          }}
        >
          BARATIE
        </h1>
      </div>
    </div>
  );
}
