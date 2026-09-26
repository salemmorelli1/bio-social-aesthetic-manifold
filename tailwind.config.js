/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/app/**/*.{ts,tsx}", "./src/components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07121b",
        surface: "#112330",
        aqua: "#70ded5",
        muted: "#9cb5c0",
      },
    },
  },
  plugins: [],
};
