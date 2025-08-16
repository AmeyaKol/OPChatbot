# AI Document Assistant

A modern, minimal Streamlit application that provides an AI-powered document editing and reference management system using Google Gemini.

## Features

- **Three-Pane Layout**: Chat (left), Document (center), and References (right)
- **AI-Powered Editing**: Use Google Gemini to edit documents based on natural language instructions
- **Smart Context**: Chatbot automatically uses document content and uploaded references as context
- **Collapsible Panes**: Adjustable layout with collapsible sections for optimal workspace
- **File Management**: Upload and manage reference documents (.txt files)
- **Dual Chat Modes**: 
  - **Discuss Mode**: Chat normally using document and references as context
  - **Update Mode**: AI automatically updates the document based on your instructions

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Git

## Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd StreamlitTrial
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory with your Google Gemini API credentials:
```env
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

## Usage

### 1. Start the Application
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### 2. Using the Application

#### Document Pane (Center)
- Type or paste your document content
- Use the "Manual Text Selection for Chat" input to copy specific text for context
- Character and word counts are displayed automatically

#### Chat Pane (Left)
- **Discuss Mode**: Chat normally with the AI using document and references as context
- **Update Mode**: AI will automatically update the document based on your instructions
- Toggle between modes using the buttons at the top of the chat pane

#### References Pane (Right)
- Upload `.txt` files using the file uploader
- View uploaded documents as scrollable cards
- Copy document previews for use in chat

### 3. AI Document Editing

1. **Switch to Update Mode**: Click the "✏️ Update" button in the chat pane
2. **Provide Instructions**: Tell the AI what changes you want (e.g., "Make this more professional", "Add a conclusion paragraph")
3. **AI Updates**: The AI will automatically modify your document based on your instructions

### 4. Context-Aware Chat

- The chatbot automatically uses:
  - Current document content
  - Uploaded reference files
  - Manually selected text (if provided)
- This ensures relevant and contextual responses

## Project Structure

```
StreamlitTrial/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── styles.css         # Custom CSS styling
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Configuration

### Customizing the AI Model
Edit the `.env` file to change the Gemini model:
```env
GEMINI_MODEL=gemini-2.0-flash-exp  # Alternative model
```

### Styling
Modify `styles.css` to customize the appearance of the application.

## Troubleshooting

### Common Issues

1. **"No module named 'google.generativeai'"**
   - Ensure you've activated your virtual environment
   - Run `pip install -r requirements.txt` again

2. **API Key Errors**
   - Verify your `.env` file exists and contains the correct API key
   - Ensure the API key has access to the specified Gemini model

3. **Streamlit Not Starting**
   - Check if port 8501 is available
   - Try `streamlit run app.py --server.port 8502` to use a different port

### Getting Help

- Check the Streamlit documentation: https://docs.streamlit.io/
- Google Gemini API documentation: https://ai.google.dev/docs
- Ensure all dependencies are properly installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini](https://ai.google.dev/)
- Icons and styling inspired by modern UI/UX practices
