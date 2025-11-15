import React from "react";

const Navbar = () => {
  return (
    <header className="bg-white shadow-md py-4 sticky top-0 z-50">
      <nav className="container mx-auto flex justify-between items-center px-4">
        <h1 className="text-2xl font-bold text-red-500">🍽️ Recipe Finder</h1>
        <ul className="hidden md:flex gap-6 text-gray-700 font-medium">
          <li className="hover:text-red-500 cursor-pointer transition">Home</li>
          <li className="hover:text-red-500 cursor-pointer transition">About</li>
          <li className="hover:text-red-500 cursor-pointer transition">Contact</li>
        </ul>
      </nav>
    </header>
  );
};

export default Navbar;
