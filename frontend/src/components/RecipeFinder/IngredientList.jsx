import { X } from 'lucide-react';

export default function IngredientList({ ingredients, handleRemoveIngredient }) {
  return (
    <div className="mt-6">
      <h3 className="font-medium mb-3" style={{ color: '#001B48' }}>
        Your Ingredients ({ingredients.length})
      </h3>
      <div className="flex flex-wrap gap-2">
        {ingredients.map((ing, idx) => (
          <span key={idx} className="inline-flex items-center gap-2 px-3 py-2 rounded-full" style={{ backgroundColor: '#E9B239', color: '#001B48' }}>
            {ing}
            <button onClick={() => handleRemoveIngredient(ing)} className="rounded-full p-1 hover:opacity-70">
              <X className="w-4 h-4" />
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}
