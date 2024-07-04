import random
import os

def generate_sentences(theme, example_sentences, num_sentences):
    sentences = random.choices(example_sentences, k=num_sentences)  # Randomly pick with replacement
    return sentences

def write_to_file(filename, themes, num_sentences_per_theme):
    with open(filename, 'w', encoding='utf-8') as file:
        # choose a random theme and generate sentences for it
        theme = random.choice(list(themes.keys()))
        example_sentences = themes[theme]
        file.write(f"--- {theme.upper()} ---\n")
        sentences = generate_sentences(theme, example_sentences, num_sentences_per_theme)
        file.write('\n'.join(sentences) + '\n\n')

def create_files(num_files, num_sentences_per_theme):
    themes = {
        'pets': [
            "Dogs often bark to express their feelings.",
            "Cats love scratching posts that mimic trees.",
            "Fish are popular pets for small apartments.",
            "Parrots can mimic human speech quite accurately.",
            "Horses require a large amount of outdoor space.",
            "Rabbits thrive on a diet rich in hay.",
            "Guinea pigs make a series of interesting vocalizations.",
            "Turtles can live for many decades with proper care.",
            "Snakes can be intriguing, though somewhat unconventional pets.",
            "Ferrets are playful animals that enjoy interacting with humans."
            # Add more actual sentences if needed
        ],
        'technology': [
            "Augmented reality offers interactive experiences.",
            "Blockchain could revolutionize digital transactions.",
            "3D printing enables rapid prototyping of designs.",
            "Renewable energy tech is crucial for sustainability.",
            "Big data analysis helps uncover hidden patterns.",
            "Drones provide a new perspective for photographers.",
            "Self-driving cars are changing the automotive industry.",
            "Smart homes integrate technology for convenience.",
            "Quantum computing promises unprecedented processing power.",
            "Machine learning algorithms improve over time."
            # Add more actual sentences if needed
        ],
        'travel': [
            "Budget travel requires careful planning and research.",
            "Cruise ships offer all-inclusive vacation experiences.",
            "Solo travel can be a rewarding challenge.",
            "Eco-tourism promotes sustainable travel practices.",
            "Historical sites attract visitors with an interest in the past.",
            "Local cuisine is a highlight of any trip.",
            "Travel bloggers often share tips and guides online.",
            "Language barriers can be overcome with technology.",
            "Packing light simplifies airport experiences.",
            "Off-season travel can offer significant savings."
            # Add more actual sentences if needed
        ],
        'cooking': [
            "Vegetarian dishes are becoming increasingly popular.",
            "Mastering the grill takes practice and patience.",
            "Pastry chefs require a precise approach to baking.",
            "Homemade pasta has a texture that’s hard to beat.",
            "Caramelization adds depth of flavor to vegetables.",
            "Meal prepping can save time during busy weeks.",
            "Sushi rolling is an art form in Japanese cuisine.",
            "Food plating requires a sense of aesthetics.",
            "Slow cooking allows flavors to develop fully.",
            "Fermentation is an ancient technique still used today."
            # Add more actual sentences if needed
        ]
        # Add more themes and sentences as needed
    }

    # Ensure the output directory exists
    output_directory = "embedding_files"
    os.makedirs(output_directory, exist_ok=True)

    # Generate and write to the specified number of files
    for i in range(num_files):
        filename = os.path.join(output_directory, f"embedding_content_{i+1}.txt")
        write_to_file(filename, themes, num_sentences_per_theme)
        print(f"Content generated and written to {filename}")

# Configurations
num_files_to_create = 100000  # 100,000 files
num_sentences_per_theme = 200  # 10 sentences per theme

# Call the function to create files
create_files(num_files_to_create, num_sentences_per_theme)