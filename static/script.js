document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("analysis-form");
    const textInput = document.getElementById("text-input");
    const channelInput = document.getElementById("channel-input");
    const dateInput = document.getElementById("date-input");

    const loader = document.getElementById("loader");
    const results = document.getElementById("results");

    const generalTable = document.getElementById("general-table");
    const nerTable = document.getElementById("ner-table");
    const thesaurusTable = document.getElementById("thesaurus-table");
    const llmSummary = document.getElementById("llm-summary");
    const messageTableBody = document.querySelector("#messageTable tbody");

    const errorMessage = document.getElementById("error-message");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        results.classList.add("hidden");
        errorMessage.classList.add("hidden");

        const text = textInput.value.trim();
        const channel = channelInput.value.trim();
        let dateStr = dateInput.value;

        if (!text) {
            showError("Мәтін енгізу міндетті!");
            return;
        }

        if (!dateStr) {
            const now = new Date();
            dateStr = now.toISOString().replace("T", " ").slice(0, 19);
        } else {
            dateStr = dateStr.replace("T", " ");
        }

        loader.classList.remove("hidden");

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                     text: text,
                     channel: channel,
                     date: dateStr
                 })
            });

            if (!response.ok) {
                throw new Error("Сервер жауап бермеді");
            }

            const report = await response.json();

            loader.classList.add("hidden");
            fillTables(report);
            results.classList.remove("hidden");

        } catch (err) {
            loader.classList.add("hidden");
            showError("❌ Серверге қосыла алмадық! FastAPI жұмыс істеп тұр ма?");
        }
    });

    function fillTables(data) {
        const a = data.analysis_report;

        generalTable.innerHTML = `
            <tr><td><b>Канал</b></td><td>${data.source_info.channel}</td></tr>
            <tr><td><b>Дата</b></td><td>${data.source_info.date}</td></tr>
            <tr><td><b>IO_TYPE</b></td><td>${a.predicted_info_operation_type}</td></tr>
            <tr><td><b>FAKE_CLAIM</b></td><td>${a.is_anomaly ? "Иә" : "Жоқ"}</td></tr>
            <tr><td><b>EMO_EVAL</b></td><td>${a.general_sentiment.label} (${a.general_sentiment.score})</td></tr>
        `;

        nerTable.innerHTML =
            "<tr><th>Сөз</th><th>Түрі</th></tr>" +
            a.named_entities_recognition
                .map(e => `<tr><td>${e.word}</td><td>${e.entity}</td></tr>`)
                .join("");

        thesaurusTable.innerHTML =
            "<tr><th>Термин</th><th>Сәйкес сөз</th><th>Түрі</th></tr>" +
            a.military_terms_analysis
                .map(t => `<tr><td>${t.term_kz}</td><td>${t.matched_alias}</td><td>${t.match_type}</td></tr>`)
                .join("");

        llmSummary.innerHTML = `
            <p><b>Қорытынды:</b> ${a.llm_expert_summary.summary}</p>
            <p><b>Қауіп деңгейі:</b> ${a.llm_expert_summary.threat_level}</p>
        `;
    }

    function addRecordToFeed(data) {
        const row = document.createElement("tr");
        const a = data.analysis_report;
        
        // Данные для клика по строке (чтобы работало "Message Analysis" при клике на новую строку)
        row.dataset.source = data.source_info.channel;
        row.dataset.date = data.source_info.date;
        row.dataset.text = data.original_text;
        row.dataset.io = a.predicted_info_operation_type;
        row.dataset.emo = a.general_sentiment.label;
        row.dataset.fake = a.is_anomaly ? "True" : "False";

        // Заполняем ячейки
        row.innerHTML = `
            <td>${data.source_info.channel}</td>
            <td>${data.source_info.date}</td>
            <td>${data.original_text.substring(0, 50) + "..."}</td>
            <td>${a.predicted_info_operation_type}</td>
            <td>${a.general_sentiment.label}</td>
            <td>${a.is_anomaly ? "True" : "False"}</td>
            <td>
                <button disabled title="Обновите страницу для удаления">♻ Жаңа</button>
            </td>
        `;

        // Добавляем обработчик клика на новую строку
        row.addEventListener("click", () => {
            document.getElementById("aSource").textContent = row.dataset.source;
            document.getElementById("aDate").textContent = row.dataset.date;
            document.getElementById("aIO").textContent = row.dataset.io;
            document.getElementById("aFake").textContent = row.dataset.fake;
            document.getElementById("aEmo").textContent = row.dataset.emo;
            document.getElementById("aText").textContent = row.dataset.text;
            document.querySelector(".analysis").scrollIntoView({ behavior: "smooth" });
        });

        // Вставляем строку в начало таблицы
        messageTableBody.insertBefore(row, messageTableBody.firstChild);
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove("hidden");
    }
});
