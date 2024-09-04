async function startGame() {
    const response = await fetch('/start');
    const data = await response.json();
    document.getElementById('game-info').innerText = JSON.stringify(data, null, 2);
}
