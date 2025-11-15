import React from 'react';
import { Trash2, ChevronRight } from 'lucide-react';

export default function SavedRecipes({ savedRecipes, onRemove, onSelect }) {
  return (
    <div style={{ 
      backgroundColor: '#ffffff', 
      minHeight: '100vh', 
      paddingBottom: '100px',
      backgroundImage: 'url(/bg.png)',
      backgroundSize: 'cover',
      backgroundPosition: window.innerWidth < 768 ? 'right center' : 'center',
      backgroundAttachment: 'fixed'
    }}>
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8 pt-4 px-2">
          <h1 className="text-5xl font-bold" style={{ color: '#001B48' }}>
            Saved Recipes
          </h1>
          <p style={{ color: '#001B48' }} className="text-lg mt-2">
            Your favorite recipes in one place
          </p>
        </div>

        {/* Saved Recipes List */}
        {savedRecipes.length > 0 ? (
          <div className="space-y-4">
            {savedRecipes.map((recipe) => (
              <div
                key={recipe.id}
                style={{ backgroundColor: '#ffffff', borderColor: '#D9D9D9' }}
                className="border-2 rounded-xl p-6 flex items-center justify-between hover:shadow-lg transition-shadow cursor-pointer"
                onClick={() => onSelect(recipe)}
              >
                {/* Left Side - Recipe Info */}
                <div className="flex items-center gap-6 flex-1">
                  {/* Recipe Image/Emoji */}
                  <div
                    style={{ backgroundColor: '#E9B239' }}
                    className="w-20 h-20 rounded-lg flex items-center justify-center text-5xl flex-shrink-0"
                  >
                    {typeof recipe.image === 'string' && recipe.image.startsWith('http') ? (
                      <img
                        src={recipe.image}
                        alt={recipe.name}
                        className="w-full h-full object-cover rounded-lg"
                      />
                    ) : (
                      recipe.image
                    )}
                  </div>

                  {/* Recipe Details */}
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold mb-2" style={{ color: '#001B48' }}>
                      {recipe.name}
                    </h3>
                    <div className="flex gap-6 mb-3">
                      <span style={{ color: '#001B48' }}>⏱️ {recipe.time}</span>
                      <span style={{ color: '#001B48' }}>📊 {recipe.difficulty}</span>
                    </div>

                    {/* Ingredient Match */}
                    <div className="flex items-center gap-3">
                      <span style={{ color: '#001B48' }} className="text-sm">
                        Ingredient Match:
                      </span>
                      <div style={{ backgroundColor: '#D9D9D9' }} className="w-32 h-2 rounded-full">
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
                      <span style={{ color: '#E9B239' }} className="font-semibold text-sm">
                        {recipe.matchedIngredients}/{recipe.totalIngredients}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Right Side - Actions */}
                <div className="flex items-center gap-4 ml-6">
                  <ChevronRight style={{ color: '#001B48' }} className="w-6 h-6" />
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(recipe.id);
                    }}
                    className="p-3 rounded-lg transition-colors"
                    style={{ backgroundColor: '#D9D9D9', color: '#001B48' }}
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <p className="text-xl" style={{ color: '#001B48' }}>
              No saved recipes yet
            </p>
            <p style={{ color: '#001B48' }} className="mt-2">
              Start exploring recipes and save your favorites!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
