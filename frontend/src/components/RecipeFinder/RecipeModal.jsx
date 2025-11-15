import { motion } from "framer-motion";

export default function RecipeModal({ recipe, onClose }) {
  if (!recipe) return null;

  return (
    <motion.div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.div
        className="bg-white rounded-2xl p-8 w-[90%] max-w-xl max-h-[90vh] overflow-y-auto shadow-xl"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 160 }}
      >
        <h2 className="text-2xl font-bold mb-4">{recipe.name}</h2>

        {recipe.image && (
          <img src={recipe.image} className="w-full rounded-xl mb-4" />
        )}

        <p className="text-gray-700 mb-2">⏱ {recipe.time}</p>
        <p className="text-gray-700 mb-2">📊 {recipe.difficulty}</p>
        <p className="text-gray-700 mb-4">🔥 {recipe.calories} kcal</p>

        <h3 className="text-lg font-semibold mb-2">Ingredients</h3>
        <ul className="list-disc ml-6 space-y-1 mb-4">
          {recipe.ingredients.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>

        <h3 className="text-lg font-semibold mb-2">Instructions</h3>
        <p className="text-gray-700 leading-relaxed whitespace-pre-line">
          {recipe.instructions}
        </p>

        <button
          onClick={onClose}
          className="mt-6 w-full py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-900"
        >
          Close
        </button>
      </motion.div>
    </motion.div>
  );
}