import json
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RecipeRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Recipe Recommender Pro")
        self.root.geometry("1100x800")

        self.configure_styles()
        self.initialize_data_files()
        self.load_all_data()
        self.init_recommendation_engine()
        self.create_ui()
        self.update_status(f"Loaded {len(self.recipes)} recipes | Ready")

    def configure_styles(self):
        """Configure custom styles for the UI"""
        style = ttk.Style()
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5')
        style.configure('TButton', padding=5)
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Success.TButton', foreground='green')
        style.configure('Danger.TButton', foreground='red')
        style.map('Success.TButton',
                  foreground=[('pressed', 'green'), ('active', 'green')],
                  background=[('pressed', '!disabled', '#d5f5d5'), ('active', '#e5f5e5')])
        style.map('Danger.TButton',
                  foreground=[('pressed', 'red'), ('active', 'red')],
                  background=[('pressed', '!disabled', '#f5d5d5'), ('active', '#f5e5e5')])

    def initialize_data_files(self):
        """Initialize data file paths"""
        self.data_files = {
            'recipes': 'recipes.json',
            'feedback': 'user_feedback.json',
            'profile': 'user_profile.json'
        }

    def load_all_data(self):
        """Load all required data files"""
        self.recipes = self.load_json_file(self.data_files['recipes'], 'recipes', [])
        self.feedback = self.load_json_file(self.data_files['feedback'], default={})
        self.profile = self.load_json_file(self.data_files['profile'], default={
            'favorite_ingredients': [],
            'liked_recipes': [],
            'disliked_recipes': [],
            'preferred_cuisines': [],
            'allergies': [],
            'cooking_skills': 'intermediate'
        })

    def load_json_file(self, filename, key=None, default=None):
        """Helper function to load JSON files"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(key, default) if key else data
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def save_json_file(self, filename, data):
        """Helper function to save JSON files"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {filename}: {str(e)}")
            return False

    def init_recommendation_engine(self):
        """Initialize the recommendation engine components"""
        self.recipe_texts = [
            ' '.join([
                ' '.join(recipe['ingredients']),
                recipe.get('cuisine', ''),
                ' '.join(k for k in recipe.get('nutrition', {}).keys())
            ])
            for recipe in self.recipes
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.recipe_texts)
        self.recipe_similarities = cosine_similarity(self.tfidf_matrix)
        self.cuisines = sorted(list(set(
            recipe.get('cuisine', 'Other') for recipe in self.recipes
        )))

    def create_ui(self):
        """Create the main user interface"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_header()
        self.create_search_panel()
        self.create_results_panel()
        self.create_status_bar()

    def create_header(self):
        """Create the header section"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=5)
        ttk.Label(
            header_frame,
            text="Smart Recipe Recommender Pro",
            style='Title.TLabel'
        ).pack(side=tk.LEFT)

    def create_search_panel(self):
        """Create the search control panel"""
        search_frame = ttk.LabelFrame(
            self.main_frame,
            text="Search & Recommendations",
            padding="10"
        )
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Ingredients:").grid(
            row=0, column=0, sticky='w', padx=5, pady=2)
        self.ingredients_entry = ttk.Entry(search_frame, width=60)
        self.ingredients_entry.grid(
            row=0, column=1, padx=5, pady=2, sticky='ew')
        self.ingredients_entry.insert(0, "chicken, onion, garlic")

        ttk.Label(search_frame, text="Cuisine:").grid(
            row=1, column=0, sticky='w', padx=5, pady=2)
        self.cuisine_var = tk.StringVar(value="All")
        self.cuisine_menu = ttk.OptionMenu(
            search_frame, self.cuisine_var, "All", *(["All"] + self.cuisines))
        self.cuisine_menu.grid(
            row=1, column=1, sticky='ew', padx=5, pady=2)

        ttk.Label(search_frame, text="Skill Level:").grid(
            row=2, column=0, sticky='w', padx=5, pady=2)
        self.skill_var = tk.StringVar(value="Any")
        self.skill_menu = ttk.OptionMenu(
            search_frame, self.skill_var, "Any",
            "Any", "Beginner", "Intermediate", "Advanced")
        self.skill_menu.grid(
            row=2, column=1, sticky='ew', padx=5, pady=2)

        btn_frame = ttk.Frame(search_frame)
        btn_frame.grid(row=3, columnspan=2, pady=10)

        ttk.Button(
            btn_frame,
            text="🔍 Search Recipes",
            command=self.search_recipes
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="💡 Get Recommendations",
            command=self.get_personalized_recommendations
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="🔄 Reset Filters",
            command=self.reset_filters
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="⭐ My Favorites",
            command=self.show_favorites
        ).pack(side=tk.LEFT, padx=5)

    def reset_filters(self):
        """Reset all filters to default values"""
        self.cuisine_var.set("All")
        self.skill_var.set("Any")
        self.ingredients_entry.delete(0, tk.END)
        self.ingredients_entry.insert(0, "chicken, onion, garlic")
        self.clear_results()
        self.update_status("Filters reset to default")

    def create_results_panel(self):
        """Create the results display panel"""
        results_frame = ttk.LabelFrame(
            self.main_frame,
            text="Recipe Results",
            padding="10"
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=('name', 'cuisine', 'score', 'ingredients', 'cook_time'),
            show='headings',
            selectmode='browse'
        )

        columns = {
            'name': ('Recipe', 200),
            'cuisine': ('Cuisine', 120),
            'score': ('Match', 80),
            'ingredients': ('Key Ingredients', 350),
            'cook_time': ('Time', 80)
        }

        for col, (text, width) in columns.items():
            self.results_tree.heading(col, text=text)
            self.results_tree.column(col, width=width, anchor='center')

        scrollbar = ttk.Scrollbar(
            results_frame,
            orient='vertical',
            command=self.results_tree.yview
        )
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side='left', fill=tk.BOTH, expand=True)
        scrollbar.pack(side='right', fill='y')
        self.results_tree.bind('<Double-1>', self.show_recipe_details)

        self.results_tree.tag_configure('liked', background='#e6f3e6')
        self.results_tree.tag_configure('disliked', background='#f3e6e6')

    def create_status_bar(self):
        """Create the status bar"""
        self.status_var = tk.StringVar()
        status_bar = ttk.Frame(self.main_frame, height=20)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(
            status_bar,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X)

    def search_recipes(self):
        """Search recipes based on ingredients and filters"""
        self.clear_results()
        ingredients = [
            i.strip().lower()
            for i in self.ingredients_entry.get().split(',')
            if i.strip()
        ]

        if not ingredients:
            messagebox.showwarning("Warning", "Please enter ingredients")
            return

        filtered_recipes = self.apply_filters()

        if not filtered_recipes:
            messagebox.showinfo("Info", "No recipes match your filters")
            return

        user_text = ' '.join(ingredients)
        user_vector = self.vectorizer.transform([user_text])
        similarities = cosine_similarity(
            user_vector,
            self.tfidf_matrix[[i for i, _ in filtered_recipes]]
        ).flatten()

        results = []
        for (idx, (i, recipe)), score in zip(enumerate(filtered_recipes), similarities):
            personalized_score = self.apply_personalization(recipe, score)
            results.append((i, recipe, personalized_score))

        self.display_results(results)

    def get_personalized_recommendations(self):
        """Get recommendations based on user profile"""
        self.clear_results()

        if not self.profile['liked_recipes']:
            messagebox.showinfo("Info", "Please like some recipes first to get recommendations")
            return

        liked_indices = [
            i for i, recipe in enumerate(self.recipes)
            if recipe['name'] in self.profile['liked_recipes']
        ]

        if not liked_indices:
            messagebox.showinfo("Info", "No liked recipes found")
            return

        similarity_scores = np.mean(self.recipe_similarities[liked_indices, :], axis=0)
        results = []
        for i, score in enumerate(similarity_scores):
            recipe = self.recipes[i]
            if self.passes_filters(recipe):
                personalized_score = self.apply_personalization(recipe, score)
                results.append((i, recipe, personalized_score))

        self.display_results(results)

    def show_favorites(self):
        """Show the user's favorite recipes"""
        self.clear_results()

        if not self.profile['liked_recipes']:
            messagebox.showinfo("Info", "You haven't liked any recipes yet")
            return

        favorites = []
        for i, recipe in enumerate(self.recipes):
            if recipe['name'] in self.profile['liked_recipes']:
                favorites.append((i, recipe, 1.0))

        favorites.reverse()
        self.display_results(favorites)
        self.update_status(f"Showing {len(favorites)} favorite recipes")

    def apply_filters(self):
        """Apply all active filters to recipes"""
        filtered = []
        for i, recipe in enumerate(self.recipes):
            if self.passes_filters(recipe):
                filtered.append((i, recipe))
        return filtered

    def passes_filters(self, recipe):
        """Check if a recipe passes all active filters"""
        cuisine_filter = self.cuisine_var.get()
        if cuisine_filter != "All" and recipe.get('cuisine') != cuisine_filter:
            return False

        skill_filter = self.skill_var.get()
        if skill_filter != "Any":
            recipe_skill = recipe.get('difficulty', 'intermediate').lower()
            if skill_filter.lower() == 'beginner' and recipe_skill not in ['beginner', 'easy']:
                return False
            elif skill_filter.lower() == 'advanced' and recipe_skill not in ['advanced', 'expert']:
                return False

        return True

    def apply_personalization(self, recipe, base_score):
        """Apply personalization boosts/penalties to a recipe score"""
        score = base_score

        if recipe['name'] in self.profile['liked_recipes']:
            score *= 1.5
        elif recipe['name'] in self.profile['disliked_recipes']:
            score *= 0.2

        recipe_ingredients = set(ing.lower() for ing in recipe['ingredients'])
        favorite_ingredients = set(ing.lower() for ing in self.profile['favorite_ingredients'])
        common_ingredients = recipe_ingredients.intersection(favorite_ingredients)

        if common_ingredients:
            score *= 1.0 + (0.1 * len(common_ingredients))

        if (recipe.get('cuisine') and
                recipe['cuisine'].lower() in [c.lower() for c in self.profile['preferred_cuisines']]):
            score *= 1.3

        return score

    def display_results(self, results):
        """Display results in the treeview"""
        results.sort(key=lambda x: x[2], reverse=True)

        for i, recipe, score in results[:100]:
            tags = []
            if recipe['name'] in self.profile['liked_recipes']:
                tags.append('liked')
            elif recipe['name'] in self.profile['disliked_recipes']:
                tags.append('disliked')

            self.results_tree.insert('', 'end',
                                     values=(
                                         recipe['name'],
                                         recipe.get('cuisine', 'Unknown'),
                                         f"{score:.2f}",
                                         ', '.join(recipe['ingredients'][:3]),
                                         recipe.get('cook_time', 'N/A')
                                     ),
                                     tags=tuple(tags)
                                     )

        self.update_status(f"Showing {min(len(results), 100)} results")

    def show_recipe_details(self, event):
        """Show detailed view of selected recipe"""
        selected = self.results_tree.selection()
        if not selected:
            return

        item = self.results_tree.item(selected[0])
        recipe_name = item['values'][0]
        recipe = next((r for r in self.recipes if r['name'] == recipe_name), None)

        if not recipe:
            return

        detail_win = tk.Toplevel(self.root)
        detail_win.title(f"Recipe: {recipe_name}")
        detail_win.geometry("900x700")

        main_frame = ttk.Frame(detail_win, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_recipe_header(main_frame, recipe)
        self.create_recipe_notebook(main_frame, recipe)
        self.create_feedback_buttons(main_frame, recipe, detail_win)

    def create_recipe_header(self, parent, recipe):
        """Create recipe header section"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            header_frame,
            text=recipe['name'],
            style='Title.TLabel'
        ).pack(side=tk.LEFT)

        meta_frame = ttk.Frame(header_frame)
        meta_frame.pack(side=tk.RIGHT)

        ttk.Label(
            meta_frame,
            text=f"Cuisine: {recipe.get('cuisine', 'Unknown')}"
        ).pack(anchor='e')

        ttk.Label(
            meta_frame,
            text=f"Difficulty: {recipe.get('difficulty', 'Intermediate')}"
        ).pack(anchor='e')

        ttk.Label(
            meta_frame,
            text=f"Cook Time: {recipe.get('cook_time', 'Not specified')}"
        ).pack(anchor='e')

    def create_recipe_notebook(self, parent, recipe):
        """Create notebook with recipe tabs"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.create_ingredients_tab(notebook, recipe)
        self.create_instructions_tab(notebook, recipe)

        if 'nutrition' in recipe:
            self.create_nutrition_tab(notebook, recipe)

    def create_ingredients_tab(self, notebook, recipe):
        """Create ingredients tab"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Ingredients")

        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, ingredient in enumerate(recipe['ingredients'], 1):
            ttk.Label(
                scrollable_frame,
                text=f"{i}. {ingredient}",
                anchor='w'
            ).pack(fill=tk.X, padx=10, pady=2)

    def create_instructions_tab(self, notebook, recipe):
        """Create instructions tab"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Instructions")

        text = tk.Text(
            tab,
            wrap='word',
            padx=10,
            pady=10,
            font=('Helvetica', 11)
        )
        scrollbar = ttk.Scrollbar(
            tab,
            orient='vertical',
            command=text.yview
        )
        text.configure(yscrollcommand=scrollbar.set)

        text.insert('1.0', recipe['instructions'])
        text.configure(state='disabled')

        text.pack(side='left', fill=tk.BOTH, expand=True)
        scrollbar.pack(side='right', fill='y')

    def create_nutrition_tab(self, notebook, recipe):
        """Create nutrition tab"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Nutrition")

        for i, (key, value) in enumerate(recipe['nutrition'].items(), 1):
            row = ttk.Frame(tab)
            row.pack(fill=tk.X, padx=10, pady=2)

            ttk.Label(
                row,
                text=key.capitalize(),
                width=20,
                anchor='w'
            ).pack(side=tk.LEFT)

            ttk.Label(
                row,
                text=str(value),
                anchor='w'
            ).pack(side=tk.LEFT)

    def create_feedback_buttons(self, parent, recipe, window):
        """Create feedback buttons"""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)

        like_text = "❤️ Liked" if recipe['name'] in self.profile['liked_recipes'] else "👍 Like"
        like_btn = ttk.Button(
            btn_frame,
            text=like_text,
            style='Success.TButton' if recipe['name'] not in self.profile['liked_recipes'] else '',
            command=lambda: self.handle_feedback(recipe, 'like', window)
        )
        like_btn.pack(side=tk.LEFT, padx=10)

        dislike_text = "💔 Disliked" if recipe['name'] in self.profile['disliked_recipes'] else "👎 Dislike"
        dislike_btn = ttk.Button(
            btn_frame,
            text=dislike_text,
            style='Danger.TButton' if recipe['name'] not in self.profile['disliked_recipes'] else '',
            command=lambda: self.handle_feedback(recipe, 'dislike', window)
        )
        dislike_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            btn_frame,
            text="Close",
            command=window.destroy
        ).pack(side=tk.RIGHT, padx=10)

    def handle_feedback(self, recipe, feedback_type, window):
        """Handle user feedback on recipes"""
        recipe_name = recipe['name']

        if recipe_name not in self.feedback:
            self.feedback[recipe_name] = {'likes': 0, 'dislikes': 0}

        if feedback_type == 'like':
            if recipe_name in self.profile['disliked_recipes']:
                self.profile['disliked_recipes'].remove(recipe_name)
                self.feedback[recipe_name]['dislikes'] = max(0, self.feedback[recipe_name]['dislikes'] - 1)

            if recipe_name not in self.profile['liked_recipes']:
                self.profile['liked_recipes'].append(recipe_name)
                self.feedback[recipe_name]['likes'] += 1

                for ing in recipe['ingredients']:
                    if ing.lower() not in [i.lower() for i in self.profile['favorite_ingredients']]:
                        self.profile['favorite_ingredients'].append(ing)

                if 'cuisine' in recipe:
                    cuisine = recipe['cuisine']
                    if cuisine.lower() not in [c.lower() for c in self.profile['preferred_cuisines']]:
                        self.profile['preferred_cuisines'].append(cuisine)
        else:
            if recipe_name in self.profile['liked_recipes']:
                self.profile['liked_recipes'].remove(recipe_name)
                self.feedback[recipe_name]['likes'] = max(0, self.feedback[recipe_name]['likes'] - 1)

            if recipe_name not in self.profile['disliked_recipes']:
                self.profile['disliked_recipes'].append(recipe_name)
                self.feedback[recipe_name]['dislikes'] += 1

        self.save_all_data()
        self.update_feedback_buttons(window, recipe)

    def save_all_data(self):
        """Save all user data to files"""
        self.save_json_file(self.data_files['feedback'], self.feedback)
        self.save_json_file(self.data_files['profile'], self.profile)

    def update_feedback_buttons(self, window, recipe):
        """Update the feedback buttons after interaction"""
        for widget in window.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        if 'Like' in child['text']:
                            if recipe['name'] in self.profile['liked_recipes']:
                                child.config(text="❤️ Liked", style='')
                            else:
                                child.config(text="👍 Like", style='Success.TButton')
                        elif 'Dislike' in child['text']:
                            if recipe['name'] in self.profile['disliked_recipes']:
                                child.config(text="💔 Disliked", style='')
                            else:
                                child.config(text="👎 Dislike", style='Danger.TButton')

        messagebox.showinfo("Feedback Saved", "Your preference has been recorded!")

    def clear_results(self):
        """Clear the results treeview"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def update_status(self, message):
        """Update the status bar"""
        self.status_var.set(message)


if __name__ == "__main__":
    root = tk.Tk()
    app = RecipeRecommenderApp(root)
    root.mainloop()