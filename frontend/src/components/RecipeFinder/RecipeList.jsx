import { useState } from "react";
import RecipeCard from "./RecipeCard";
import RecipeModal from "./RecipeModal";

export default function RecipeList({ recipes }) {
  const [selected, setSelected] = useState(null);

  return (
    <>
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredRecipes.map(recipe => (
            <RecipeCard
            key={recipe.id}
            recipe={recipe}
            onSelect={onSelect}   // <-- ADD THIS
        />
        ))}
      </div>

      <RecipeModal recipe={selected} onClose={() => setSelected(null)} />
    </>
  );
}
