Imitune: Your Voice is the Search Bar
Searching for sound effects and samples is tedious! Enough of digging into folders of sound packs or searching random websites - just imitate what you're looking for with your voice!

✨ Inspiration
For every music producer, game developer, or video editor, there's a universally shared pain point: the endless hunt for that "perfect" sound. The one you can hear in your head—a specific "whoosh," a "thud," or a "bleep"—but can't find by digging through countless folders or typing keyword after keyword online.

We believed the solution was within us. The person who knows the sound best is you, and the best tool to express it is your voice. This idea was solidified as we developed our award-winning "Query-by-Vocal Imitation" model at the AIMLA Challenge. We asked: why can't we take this lab-level technology and apply it to solve a real-world problem for creators? Imitune was born from that question.

🎯 What it does
Imitune is a desktop app that turns your voice into an audio search engine. It's incredibly simple:

Imitate: Open the app and imitate the sound you're looking for into your microphone.

Discover: Imitune instantly analyzes your voice and finds the most similar sounds from a massive library.

Improve: Provide feedback on the results to help us train the model. This feedback provides valuable data for our future research to continually improve the search model.

No more typing complex phrases. Just make a "whoosh" sound with your voice. Imitune provides the most intuitive way to bring the sounds from your imagination into reality, all without breaking your creative flow.

🛠️ How we built it
Imitune combines the power of local AI processing with cloud-based, high-speed search.

The AI Model: We converted our award-winning, PyTorch-based model into the ONNX format. This makes the model portable, lightweight, and fast enough to run on any platform.

The Desktop App: Using Electron, we developed a cross-platform app for both macOS and Windows. The app runs the ONNX model directly on your machine using onnxruntime-node, ensuring privacy and instant analysis without network latency.

The Similarity Search Backend: We indexed tens of thousands of sound embeddings in a Pinecone vector database. When the app generates a vector from your voice, it queries our serverless API on Vercel, which fetches the most similar results from Pinecone in milliseconds.

🚧 Challenges we ran into
(To be filled in after development)

🏆 Accomplishments that we're proud of
(To be filled in after development)

📚 What we learned
(To be filled in after development)

🚀 What's next for Imitune
(To be filled in after development)

## 🛠️ Built With

### Languages
- JavaScript (ES6+)
- Python (For original ML model development)
- HTML5
- CSS3

### Frameworks & Runtimes
- **Electron**: For cross-platform (macOS, Windows) desktop application development  
- **Node.js**: As the backend runtime for the Electron application  

### AI & Machine Learning
- **ONNX (Open Neural Network Exchange)**: For making our deep learning model portable and efficient  
- **ONNX Runtime**: To execute the model locally on the user's machine with high performance  
- **PyTorch**: The framework used to initially train our award-winning model  

### Backend & Cloud Services
- **Vercel**: For hosting our serverless backend API functions  
- **Pinecone**: As our high-performance vector database for real-time similarity search  
- **Vercel Blob**: For storing user-submitted audio query files  
- **Vercel KV**: For storing metadata and user feedback  

### Styling
- **Tailwind CSS**: For rapidly building the modern user interface of our application

## Run or Build
```
npm start
npm run build:mac
npm run build:win
```