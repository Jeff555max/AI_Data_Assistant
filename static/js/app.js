const setupComposer = () => {
    const textarea = document.getElementById("message-text");
    const fileInput = document.getElementById("data-file");
    const filePill = document.getElementById("selected-file-pill");
    const thread = document.getElementById("chat-thread");

    if (textarea) {
        const resize = () => {
            textarea.style.height = "auto";
            textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
        };

        resize();
        textarea.addEventListener("input", resize);

        document.querySelectorAll("[data-prompt]").forEach((button) => {
            button.addEventListener("click", () => {
                textarea.value = button.dataset.prompt || "";
                textarea.focus();
                resize();
            });
        });
    }

    if (fileInput && filePill) {
        const syncFilePill = () => {
            const file = fileInput.files?.[0];
            if (!file) {
                filePill.hidden = true;
                filePill.textContent = "";
                return;
            }

            filePill.hidden = false;
            filePill.textContent = `Выбрано: ${file.name}`;
        };

        fileInput.addEventListener("change", syncFilePill);
        syncFilePill();
    }

    if (thread) {
        thread.scrollTop = thread.scrollHeight;
    }
};

const toggleLoader = (show) => {
    const globalLoader = document.getElementById("global-loader");
    if (!globalLoader) {
        return;
    }
    globalLoader.style.display = show ? "inline-flex" : "none";
    globalLoader.setAttribute("aria-hidden", show ? "false" : "true");
};

const toggleProcessingPanels = (show) => {
    document.querySelectorAll("[data-processing-panel]").forEach((panel) => {
        panel.classList.toggle("is-visible", show);
    });
};

const processingProgress = (() => {
    let progress = 0;
    let timerId = null;
    let hideTimerId = null;

    const elements = () => ({
        loader: document.getElementById("global-loader"),
        title: document.getElementById("global-loader-title"),
        percent: document.getElementById("global-loader-percent"),
        bar: document.getElementById("global-loader-bar"),
        inlineTitles: document.querySelectorAll("[data-progress-title]"),
        inlinePercents: document.querySelectorAll("[data-progress-percent]"),
        inlineBars: document.querySelectorAll("[data-progress-bar]"),
    });

    const setProgress = (nextProgress, titleText) => {
        const refs = elements();
        progress = Math.max(progress, Math.min(100, Math.round(nextProgress)));

        if (refs.title && titleText) {
            refs.title.textContent = titleText;
        }
        if (refs.percent) {
            refs.percent.textContent = `${progress}%`;
        }
        if (refs.bar) {
            refs.bar.style.width = `${progress}%`;
        }

        refs.inlineTitles.forEach((title) => {
            if (titleText) {
                title.textContent = titleText;
            }
        });
        refs.inlinePercents.forEach((percent) => {
            percent.textContent = `${progress}%`;
        });
        refs.inlineBars.forEach((bar) => {
            bar.style.width = `${progress}%`;
        });
    };

    const hasSelectedFile = (source) => {
        const form = source?.closest?.("form");
        const input = form?.querySelector?.("input[type='file']");
        return Boolean(input?.files?.length);
    };

    const processingLabel = (source, fallback) => {
        const form = source?.closest?.("form");
        return source?.dataset?.processingLabel || form?.dataset?.processingLabel || fallback;
    };

    const start = (source, titleText) => {
        clearTimeout(hideTimerId);
        clearInterval(timerId);
        progress = 0;
        toggleLoader(true);
        toggleProcessingPanels(true);
        setProgress(
            4,
            processingLabel(
                source,
                titleText || (hasSelectedFile(source) ? "Загружаем файл…" : "Идёт обработка и анализ, пожалуйста подождите…"),
            ),
        );

        timerId = window.setInterval(() => {
            const limit = progress < 70 ? 70 : 95;
            const step = progress < 70 ? 8 : 3;
            const label = progress < 65 ? "Загружаем и читаем файл…" : "Идёт обработка и анализ, пожалуйста подождите…";
            if (progress < limit) {
                setProgress(Math.min(limit, progress + step), label);
            }
        }, 500);
    };

    const upload = (event) => {
        const total = event.detail?.total;
        const loaded = event.detail?.loaded;
        if (!total || !loaded) {
            return;
        }

        const uploadProgress = Math.min(70, Math.round((loaded / total) * 70));
        const label = uploadProgress >= 70 ? "Файл загружен. Идёт анализ…" : "Загружаем файл…";
        setProgress(uploadProgress, label);
    };

    const finish = (success = true) => {
        clearInterval(timerId);
        setProgress(100, success ? "Готово. Обновляем результат…" : "Не удалось завершить обработку");
        hideTimerId = window.setTimeout(() => {
            toggleLoader(false);
            toggleProcessingPanels(false);
            setProgress(0, "Готово к обработке. Запустите действие.");
        }, 700);
    };

    return { start, upload, finish };
})();

document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement) || !form.matches("[data-progress-form], .composer")) {
        return;
    }

    processingProgress.start(form);
}, true);

document.addEventListener("click", (event) => {
    const button = event.target.closest?.("button[data-processing-label]");
    if (!button || button.disabled) {
        return;
    }

    processingProgress.start(button, button.dataset.processingLabel);
}, true);

document.addEventListener("DOMContentLoaded", () => {
    setupComposer();

    document.body.addEventListener("htmx:beforeRequest", (event) => processingProgress.start(event.detail?.elt));
    document.body.addEventListener("htmx:xhr:progress", (event) => processingProgress.upload(event));
    document.body.addEventListener("htmx:afterRequest", (event) => {
        const successful = event.detail?.successful !== false;
        processingProgress.finish(successful);
    });
    document.body.addEventListener("htmx:sendError", () => processingProgress.finish(false));
    document.body.addEventListener("htmx:responseError", () => processingProgress.finish(false));
    document.body.addEventListener("htmx:afterSwap", () => {
        setupComposer();
    });
});
