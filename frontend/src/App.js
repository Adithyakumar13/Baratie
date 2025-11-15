import { useState, useEffect } from 'react';
import RecipeFinder from "./components/RecipeFinder/RecipeFinder";
import LoadingPage from "./components/LoadingPage";

export default function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Auto-hide loading page after 2s
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <LoadingPage onLoadingComplete={() => setIsLoading(false)} />;
  }

  return <RecipeFinder />;
}

