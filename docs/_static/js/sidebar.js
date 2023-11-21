const toc = [
    { header: "Overview", toc: [
        { title: "RedisVL", path: "/index.html" },
        { title: "Install", path: "/overview/installation.html" },
        { title: "Quickstart", path: "/overview/quickstart.html" },
        { title: "CLI", path: "/overview/cli.html" },
    ]},
    { header: "User Guide", toc: [
        { title: "Getting Started", path: "/user_guide/getting_started_01.html" },
        { title: "Query and Filter", path: "/user_guide/hybrid_queries_02.html" },
        { title: "Vectorizers", path: "/user_guide/vectorizers_04.html" },
        { title: "Semantic Caching", path: "/user_guide/llmcache_03.html" },
        { title: "JSON Storage", path: "/user_guide/hash_vs_json_05.html" }

    ]},
    { header: "API", toc: [
        { title: "Index", path: "/api/searchindex.html" },
        { title: "Query", path: "/api/query.html" },
        { title: "Filter", path: "/api/filter.html" },
        { title: "Vectorizers", path: "/api/vectorizer.html" },
        { title: "LLMCache", path: "/api/cache.html" }
    ]},
    { header: "Concepts", toc: [
        { title: "Vector Embeddings", path: "/concepts/vectors.html" },
        { title: "HNSW vs Flat", path: "/concepts/vector_index.html" },
        { title: "Distance Metrics", path: "/concepts/distance.html" }
]}
];

document.addEventListener('DOMContentLoaded', function() {
    buildSidebar(toc);
});

function getBasePath() {
    console.log(window.location.origin);
    return window.location.origin
}

function buildSidebar(toc) {
    let tocElement = document.getElementById('toc');
    let base = getBasePath();

    toc.forEach(section => {
        // Create and append the header
        let header = document.createElement('h4');
        header.textContent = section.header;
        tocElement.appendChild(header);

        // Create a sublist for the nested TOC items
        let sublist = document.createElement('ul');

        section.toc.forEach(item => {
            let li = document.createElement('li');

            // Create an h5 element for the title
            let title = document.createElement('h6');
            title.style.margin = '0'; // Optional: Set margin to 0 to compact the list

            // Create a link for the path
            let a = document.createElement('a');
            a.textContent = item.title;
            a.href = base + item.path;

            // Append the link to the h5 element and then the h5 element to the list item
            title.appendChild(a);
            li.appendChild(title);
            sublist.appendChild(li);
        });

        tocElement.appendChild(sublist);
    });
}