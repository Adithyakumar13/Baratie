import { motion } from "framer-motion";
import { Bookmark } from "lucide-react";
import { useState } from "react";

export default function RecipeCard({ recipe, onSelect, onSave, isSaved = false }) {
  const [saved, setSaved] = useState(isSaved);

  const handleSave = (e) => {
    e.stopPropagation();
    setSaved(!saved);
    if (onSave) {
      onSave(recipe);
    }
  };

  return (
    <motion.div
      layout
      whileHover={{ scale: 1.03 }}
      transition={{ type: "tween", duration: 0.3 }}
      onClick={() => onSelect(recipe)}
      style={{ backgroundColor: '#ffffff', borderColor: '#D9D9D9' }}
      className="cursor-pointer rounded-xl border-2 overflow-hidden hover:shadow-xl"
    >
      <div className="p-6">
        {/* Save Button */}
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-bold text-xl" style={{ color: '#001B48' }}>{recipe.name}</h3>
          <button
            onClick={handleSave}
            className="p-2 rounded-lg transition-colors"
            style={{
              backgroundColor: saved ? '#E9B239' : '#D9D9D9',
              color: '#001B48',
            }}
          >
            <Bookmark className="w-5 h-5" fill={saved ? 'currentColor' : 'none'} />
          </button>
        </div>

        {/* time + difficulty */}
        <div className="flex items-center gap-4 text-sm mb-4" style={{ color: '#001B48' }}>
          <span>⏱️ {recipe.time}</span>
          <span>📊 {recipe.difficulty}</span>
        </div>

        {/* Ingredient Match */}
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1" style={{ color: '#001B48' }}>
            <span>Ingredient Match</span>
            <span className="font-semibold" style={{ color: '#E9B239' }}>
              {recipe.matchedIngredients}/{recipe.totalIngredients}
            </span>
          </div>
          <div style={{ backgroundColor: '#D9D9D9' }} className="w-full h-2 rounded-full">
            <div
              style={{
                backgroundColor: '#E9B239',
                width: `${
                  (recipe.matchedIngredients / recipe.totalIngredients) * 100
                }%`,
              }}
              className="h-2 rounded-full"
            />
          </div>
        </div>

        <button style={{ backgroundColor: '#E9B239', color: '#001B48' }} className="w-full py-3 rounded-xl font-medium hover:shadow-lg">
          View Recipe
        </button>
      </div>
    </motion.div>
  );
}
