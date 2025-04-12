hero_logo = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    body {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: #fff;
        margin: 0;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        flex-direction: column;
    }

    .book {
        width: 50px;
        height: 60px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .book::before, .book::after {
        content: '';
        width: 25px;
        height: 60px;
        background: #00d4c0;
        position: absolute;
        border-radius: 5px;
        border: 3px solid #0066ff;
    }

    .book::before {
        left: 0;
        transform: skewY(-10deg);
    }

    .book::after {
        right: 0;
        transform: skewY(10deg);
    }

    .text {
        font-family: 'Pacifico', cursive;
        font-size: 48px;
        color: var(--text-color);
        margin-top: 20px;
    }

    .subtitle {
        font-size: 14px;
        color: var(--text-color);
        text-align: center;
        margin-top: 5px;
    }
</style>

<script>
    const setTheme = (theme) => {
        document.documentElement.style.setProperty('--text-color', theme === 'dark' ? '#fff' : '#333');
    }

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                const theme = document.body.classList.contains('dark') ? 'dark' : 'light';
                setTheme(theme);
            }
        });
    });

    observer.observe(document.body, { attributes: true });

    // Initial theme set
    const initialTheme = document.body.classList.contains('dark') ? 'dark' : 'light';
    setTheme(initialTheme);
</script>

<div class="logo-container">
    <div class="book"></div>
    <div class="text">PadhAI</div>
</div>
<div class="subtitle">REVOLUTIONIZING EDUCATION, ONE LESSON AT A TIME</div>
'''

sidebar_logo = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    body {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: #fff;
        margin: 0;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        flex-direction: column;
    }

    .book {
        width: 50px;
        height: 60px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .book::before, .book::after {
        content: '';
        width: 25px;
        height: 60px;
        background: #00d4c0;
        position: absolute;
        border-radius: 5px;
        border: 3px solid #0066ff;
    }

    .book::before {
        left: 0;
        transform: skewY(-10deg);
    }

    .book::after {
        right: 0;
        transform: skewY(10deg);
    }

    .text {
        font-family: 'Pacifico', cursive;
        font-size: 48px;
        color: var(--text-color);
        margin-top: 20px;
    }

    .subtitle {
        font-size: 14px;
        color: var(--text-color);
        text-align: center;
        margin-top: 5px;
    }
</style>

<script>
    const setTheme = (theme) => {
        document.documentElement.style.setProperty('--text-color', theme === 'dark' ? '#fff' : '#333');
    }

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                const theme = document.body.classList.contains('dark') ? 'dark' : 'light';
                setTheme(theme);
            }
        });
    });

    observer.observe(document.body, { attributes: true });

    // Initial theme set
    const initialTheme = document.body.classList.contains('dark') ? 'dark' : 'light';
    setTheme(initialTheme);
</script>

<div class="logo-container">
    <div class="book"></div>
    <div class="text">PadhAI</div>
</div>
'''