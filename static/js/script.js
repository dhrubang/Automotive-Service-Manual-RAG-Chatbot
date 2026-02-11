document.addEventListener('DOMContentLoaded', () => {
    addMessage('assistant', "Hello! 👋 I'm your Service Manual Assistant.\nAsk about torque specs, procedures, capacities, fluid types, etc.");
});

const chatHistory = document.getElementById('chat-history');
const chatForm   = document.getElementById('chat-form');
const queryInput = document.getElementById('query');

function addMessage(role, content = '', isTyping = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);

    if (isTyping) {
        msgDiv.id = 'typing-message';
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = content;
    msgDiv.appendChild(textDiv);

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv;
}

function typeWriter(element, text, speed = 30) {
    let i = 0;
    const textDiv = element.querySelector('.message-text');
    textDiv.textContent = '';
    const interval = setInterval(() => {
        if (i < text.length) {
            textDiv.textContent += text.charAt(i++);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        } else {
            clearInterval(interval);
            element.removeAttribute('id');
        }
    }, speed);
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    if (!query) return;

    addMessage('user', query);

    queryInput.value = '';

    const typingMsg = addMessage('assistant', '', true);

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ query })
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const data = await res.json();

        typingMsg.remove();
        const aiMsg = addMessage('assistant', '', true);
        typeWriter(aiMsg, data.response || "[No response received]");

    } catch (err) {
        console.error(err);
        typingMsg.remove();
        addMessage('assistant', "Sorry — something went wrong. Please try again.");
    }
});

document.getElementById('new-chat-btn').addEventListener('click', () => {
    if (confirm("Start a fresh conversation?")) {
        chatHistory.innerHTML = '';
        fetch('/new_chat', { method: 'POST' });
        addMessage('assistant', "New conversation started!\nHow can I help you today?");
    }
});