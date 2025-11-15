import { Camera } from 'lucide-react';
import { useRef } from 'react';

export default function IngredientInput({ inputValue, setInputValue, handleAddIngredient, onImageDetect }) {
  const fileInputRef = useRef(null);

  const handleCameraClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      setTimeout(() => {
        onImageDetect(['tomato', 'onion', 'garlic', 'pasta'], rd.result);
      }, 1500);
    };
    rd.readAsDataURL(file);
  };

  return (
    <div>
      <label className="block font-medium mb-3" style={{ color: '#001B48' }}>Add your ingredients</label>
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddIngredient()}
            placeholder="e.g., tomatoes, pasta, garlic..."
            className="w-full px-4 py-3 pr-10 border-2 rounded-xl focus:outline-none"
            style={{ borderColor: '#D9D9D9', color: '#001B48' }}
          />
          <button
            onClick={handleCameraClick}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 p-1 rounded-lg transition-colors hover:opacity-70"
            style={{ color: '#001B48' }}
          >
            <Camera className="w-5 h-5" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
        <button onClick={handleAddIngredient} className="px-6 py-3 rounded-xl hover:shadow-lg font-medium" style={{ backgroundColor: '#E9B239', color: '#001B48' }}>
          Add
        </button>
      </div>
    </div>
  );
}
