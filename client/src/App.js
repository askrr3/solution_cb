import React, { useState } from 'react';
import axios from 'axios';

function Chatbot() {
    const [question, setQuestion] = useState('');
    const [chatHistory, setChatHistory] = useState([]);

    const askQuestion = async () => {
        try {
            // Add the user's question to the chat history
            setChatHistory(prevHistory => [
                ...prevHistory,
                { type: 'user', text: question },
            ]);

            const response = await axios.post('http://localhost:8082/chat', { question });
            const answer = response?.data?.answer;

            // Add the API response to the chat history
            setChatHistory(prevHistory => [
                ...prevHistory,
                { type: 'api', text: answer },
            ]);

            // Clear the input field
            setQuestion('');
        } catch (error) {
            console.error(error);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            askQuestion();
        }
    };

    return (
        <div style={{ maxWidth: '550px', margin: '0 auto' }}>
            <div style={{ border: '1px solid #ccc', padding: '10px', minHeight: '200px', borderRadius: '8px', marginTop:'50px' }}>
                {chatHistory.map((entry, index) => (
                    <div
                        key={index}
                        style={{
                            margin: '5px 0',
                            padding: '8px',
                            borderRadius: '8px',
                            border: '1px solid #ddd',
                            background: entry.type === 'user' ? '#e6e6e6' : '#b3d9ff',
                            textAlign: entry.type === 'user' ? 'left' : 'right',
                            fontSize: '12px',
                        }}
                    >
                        {entry.text}
                    </div>
                ))}
            </div>
            <input
                type="text"
                value={question}
                onChange={e => setQuestion(e.target.value)}
                placeholder="Ask a question..."
                style={{
                    width: '97%',
                    padding: '8px',
                    marginTop: '10px',
                    borderRadius: '4px',
                    border: '1px solid #ccc',
                    fontSize: '12px',
                }}
                onKeyPress={handleKeyPress} // Add event listener for Enter key
            />
            <button
                onClick={askQuestion}
                style={{
                    marginTop: '10px',
                    padding: '8px 16px',
                    background: '#007bff',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '12px',
                }}
            >
                Ask
            </button>
        </div>
    );
}

export default Chatbot;
