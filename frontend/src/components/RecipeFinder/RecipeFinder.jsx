import React, { useState } from 'react';
import { Sparkles, Search, Home, Bookmark } from 'lucide-react';
import InputModeToggle from './InputModeToggle';
import IngredientInput from './IngredientInput';
import ImageUpload from './ImageUpload';
import IngredientList from './IngredientList';
import RecipeCard from './RecipeCard';
import SavedRecipes from './SavedRecipes';
import mockRecipes from './mockRecipes';

export default function RecipeFinder() {
  const [inputMode, setInputMode] = useState('text');
  const [ingredients, setIngredients] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [recipes, setRecipes] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [savedRecipes, setSavedRecipes] = useState([
    mockRecipes[0],
    mockRecipes[1],
  ]);
  const [currentScreen, setCurrentScreen] = useState('home');

  const handleAddIngredient = () => {
    if (inputValue.trim() && !ingredients.includes(inputValue.trim().toLowerCase())) {
      setIngredients([...ingredients, inputValue.trim().toLowerCase()]);
      setInputValue('');
    }
  };

  const handleRemoveIngredient = (ing) => {
    setIngredients(ingredients.filter(i => i !== ing));
  };

  const handleImageUpload = (ingList, img) => {
    setUploadedImage(img);
    setIngredients(ingList);
  };

  const handleSearch = () => {
    setIsSearching(true);
    setTimeout(() => {
      setRecipes(mockRecipes);
      setIsSearching(false);
    }, 1000);
  };

  const handleRemoveSavedRecipe = (recipeId) => {
    setSavedRecipes(savedRecipes.filter(recipe => recipe.id !== recipeId));
  };

  const handleSaveRecipe = (recipe) => {
    if (!savedRecipes.find(r => r.id === recipe.id)) {
      setSavedRecipes([...savedRecipes, recipe]);
    }
  };

  if (currentScreen === 'saved') {
    return (
      <>
        <SavedRecipes 
          savedRecipes={savedRecipes} 
          onRemove={handleRemoveSavedRecipe}
          onSelect={setSelectedRecipe}
        />
        {/* Bottom Navigation Bar */}
        <div style={{ backgroundColor: '#ffffff', borderTopColor: '#D9D9D9' }} className="fixed bottom-0 left-0 right-0 border-t-2 shadow-2xl">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            {/* Home Button */}
            <button 
              onClick={() => setCurrentScreen('home')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors" style={{ color: '#001B48' }}>
              <Home className="w-6 h-6" />
              <span className="font-medium">Home</span>
            </button>

            {/* Search Button */}
            <button style={{ backgroundColor: '#E9B239', color: '#001B48' }} className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium hover:shadow-lg transition-all">
              <Search className="w-6 h-6" />
              <span>Search</span>
            </button>

            {/* Saved Recipes Button */}
            <button 
              onClick={() => setCurrentScreen('saved')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors" style={{ color: '#001B48' }}>
              <Bookmark className="w-6 h-6" />
              <span className="font-medium">Saved</span>
            </button>
          </div>
        </div>
      </>
    );
  }

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
          <div className="inline-flex items-center gap-2 mb-3">
            <Sparkles className="w-10 h-10" style={{ color: '#E9B239' }} />
            <h1 className="text-5xl font-bold" style={{ color: '#001B48' }}>
              Baratie
            </h1>
          </div>
          <p style={{ color: '#001B48' }} className="text-lg">
            Upload a photo or list your ingredients to discover amazing recipes
          </p>
        </div>

        {/* Input Mode Toggle */}
        <InputModeToggle inputMode={inputMode} setInputMode={setInputMode} />

        {/* Input Section */}
        <div style={{ backgroundColor: '#ffffff', borderColor: '#D9D9D9' }} className="rounded-2xl border-2 shadow-xl p-8 mb-8">
          {inputMode === 'text' ? (
            <IngredientInput
              inputValue={inputValue}
              setInputValue={setInputValue}
              handleAddIngredient={handleAddIngredient}
              onImageDetect={handleImageUpload}
            />
          ) : (
            <ImageUpload onDetect={handleImageUpload} uploadedImage={uploadedImage} />
          )}

          {/* Ingredient List */}
          {ingredients.length > 0 && (
            <IngredientList ingredients={ingredients} handleRemoveIngredient={handleRemoveIngredient} />
          )}

          {/* Search Button */}
          {ingredients.length > 0 && (
            <button
              onClick={handleSearch}
              disabled={isSearching}
              style={{ backgroundColor: '#E9B239', color: '#001B48' }}
              className="w-full mt-6 py-4 rounded-xl font-bold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isSearching ? (
                <div className="w-5 h-5 border-3 border-current border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  Find Recipes
                </>
              )}
            </button>
          )}
        </div>

        {/* Results Section with Side Panel */}
        {recipes.length > 0 && (
          <div className="flex gap-6">
            {/* Recipes Grid - Left Side */}
            <div className="flex-1">
              <h2 className="text-3xl font-bold mb-6" style={{ color: '#001B48' }}>Suggested Recipes</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {recipes.map((recipe) => (
                  <RecipeCard recipe={recipe} key={recipe.id} onSelect={setSelectedRecipe} />
                ))}
              </div>
            </div>

            {/* Recipe Details - Right Side */}
            {selectedRecipe && (
              <div style={{ backgroundColor: '#ffffff', borderColor: '#D9D9D9', position: 'sticky', top: '92px' }} className="w-96 rounded-2xl border-2 shadow-xl p-8 h-fit">
                {/* Close Button */}
                <button
                  onClick={() => setSelectedRecipe(null)}
                  style={{ color: '#001B48' }}
                  className="float-right text-2xl font-bold mb-2 hover:opacity-70"
                >
                  ×
                </button>

                <h2 className="text-2xl font-bold mb-4 clear-right" style={{ color: '#001B48' }}>{selectedRecipe.name}</h2>

                {/* Recipe Info */}
                <div className="flex gap-4 mb-4 text-sm" style={{ color: '#001B48' }}>
                  <span>⏱️ {selectedRecipe.time}</span>
                  <span>📊 {selectedRecipe.difficulty}</span>
                </div>

                {/* Ingredients */}
                <div style={{ backgroundColor: '#E9B239' }} className="mb-6 p-4 rounded-xl shadow-lg">
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#001B48' }}>Ingredients</h3>
                  <ul className="list-disc list-inside space-y-2">
                    {selectedRecipe.ingredients.map((ingredient, index) => (
                      <li key={index} className="text-sm" style={{ color: '#001B48' }}>{ingredient}</li>
                    ))}
                  </ul>
                </div>

                {/* Instructions */}
                <div style={{ backgroundColor: '#D9D9D9' }} className="p-4 rounded-xl shadow-lg">
                  <h3 className="text-lg font-bold mb-3" style={{ color: '#001B48' }}>Instructions</h3>
                  <ol className="list-decimal list-inside space-y-2">
                    {selectedRecipe.instructions.map((instruction, index) => (
                      <li key={index} className="text-sm" style={{ color: '#001B48' }}>{instruction}</li>
                    ))}
                  </ol>
                </div>
              </div>
            )}
          </div>
        )}

      </div>

      {/* Bottom Navigation Bar */}
      <div style={{ backgroundColor: '#ffffff', borderTopColor: '#D9D9D9' }} className="fixed bottom-0 left-0 right-0 border-t-2 shadow-2xl">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Home Button */}
          <button 
            onClick={() => setCurrentScreen('home')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors" style={{ color: '#001B48' }}>
            <Home className="w-6 h-6" />
            <span className="font-medium">Home</span>
          </button>

          {/* Search Button */}
          <button style={{ backgroundColor: '#E9B239', color: '#001B48' }} className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium hover:shadow-lg transition-all">
            <Search className="w-6 h-6" />
            <span>Search</span>
          </button>

          {/* Saved Recipes Button */}
          <button 
            onClick={() => setCurrentScreen('saved')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors" style={{ color: '#001B48' }}>
            <Bookmark className="w-6 h-6" />
            <span className="font-medium">Saved</span>
          </button>
        </div>
      </div>
    </div>
  );
}
