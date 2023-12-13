document$.subscribe(({ body }) => {
    renderMathInElement(body, {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true }
        ],
        macros: {
            "\\norm": "\\left\\lVert #1 \\right\\rVert",
            "\\bf": "\\mathbf"
        }
    })
})