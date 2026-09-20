SRP project:
Socially relevant project


The Foodie app acts as an intelligent, AI-powered food assistant that combines computer vision, conversational AI, and health management into a single platform. Users can upload photos of dishes like pizza or potato roast to instantly identify meals using YOLO and OpenAI's CLIP models, while managing their accounts and saving custom chat histories through dedicated user routes. Beyond simple meal tracking, the app features an integrated AI advisor—powered by local LLMs via AnythingLLM—that dynamically analyzes health queries (such as checking dietary restrictions for conditions like Hepatitis) to recommend safe, tailored food options. Secured with an obscured admin dashboard for managing users and media, Foodie bridges the gap between smart visual dish recognition and personalized health-conscious meal planning.

The workflow begins when a user logs in through the authentication portal (login.urls / user.urls) and either submits a text query or uploads a meal photo (such as pizza or potato roast to media/chat_images/). If an image is provided, the backend leverages YOLO and OpenAI's CLIP to recognize the dish, while text queries—such as asking about safe diets for health conditions like Hepatitis—are routed through the /start_chat/ endpoint to a local AnythingLLM engine for tailored medical food advice. Finally, the app displays the analysis on the interface and calls /save_chat_history/ to store the interaction in the database, allowing users to track their diet and chat records seamlessly.

clip:
CLIP is a multimodal neural network trained on a massive dataset of image-text pairs. Unlike traditional models trained to recognize a fixed set of predefined categories, CLIP learns visual concepts and natural language jointly.  CLIP (and similar vision-language models like Apple's recent models or multimodal architectures) takes both images and text, maps them into the same mathematical space, and compares them.

Anythingllm:
AnythingLLM is an all-in-one, privacy-focused AI application designed to make local model deployment, document chunking, and Retrieval-Augmented Generation (RAG) accessible without complex coding. It allows users to organize files, web links, and data sources into isolated "workspaces" while remaining completely agnostic to model providers—seamlessly supporting local runners like Ollama alongside cloud APIs such as OpenAI or Anthropic. Beyond simple document chatting, the platform includes built-in multi-user management (via Docker), vector database handling, and autonomous AI agents capable of executing multi-step workflows.

