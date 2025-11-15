const mockRecipes = [
  {
    id: 1,
    name: "Classic Tomato Pasta",
    time: "20 mins",
    difficulty: "Easy",
    matchedIngredients: 4,
    totalIngredients: 6,
    image: "🍝",
    calories: 450,
    protein: 15,
    ingredients: ["2 cups pasta", "400g tomatoes", "3 cloves garlic", "2 tbsp olive oil", "Salt and pepper", "Fresh basil"],
    instructions: ["Boil pasta according to package directions", "Heat olive oil and sauté garlic", "Add tomatoes and simmer for 10 minutes", "Drain pasta and mix with sauce", "Top with fresh basil and serve"]
  },
  {
    id: 2,
    name: "Fresh Garden Salad",
    time: "10 mins",
    difficulty: "Easy",
    matchedIngredients: 5,
    totalIngredients: 5,
    image: "🥗",
    calories: 180,
    protein: 8,
    ingredients: ["2 cups mixed greens", "1 cucumber", "2 tomatoes", "1 red onion", "Olive oil dressing"],
    instructions: ["Wash and dry the greens", "Chop cucumber and tomatoes", "Slice the red onion thinly", "Combine all vegetables in a bowl", "Drizzle with olive oil dressing and toss"]
  },
  {
    id: 3,
    name: "Veggie Stir Fry",
    time: "25 mins",
    difficulty: "Medium",
    matchedIngredients: 3,
    totalIngredients: 8,
    image: "🍲",
    calories: 280,
    protein: 12,
    ingredients: ["2 cups broccoli", "1 bell pepper", "1 cup mushrooms", "3 cloves garlic", "2 tbsp soy sauce", "1 tbsp sesame oil", "Ginger", "Rice for serving"],
    instructions: ["Heat sesame oil in a wok or large pan", "Add garlic and ginger, stir-fry for 1 minute", "Add harder vegetables first (broccoli, carrots)", "After 3 minutes add softer vegetables", "Pour soy sauce and stir-fry until cooked", "Serve over rice"]
  }
];

export default mockRecipes;
