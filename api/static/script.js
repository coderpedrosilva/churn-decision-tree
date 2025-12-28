async function carregarClientes(somenteRisco = false) {

    const resp = await fetch(`/clientes?somente_risco=${somenteRisco}`);
    const clientes = await resp.json();

    const tabela = document.getElementById("tabela");
    tabela.innerHTML = "";

    clientes.forEach(c => {
        tabela.innerHTML += `
          <tr>
            <td>${c.cliente}</td>
            <td>${(c.probabilidade * 100).toFixed(0)}%</td>
            <td>${c.classe}</td>
            <td><button onclick='verDetalhes(${JSON.stringify(c)})'>Ver</button></td>
          </tr>
        `;
    });
}

function verDetalhes(cliente) {
    let html = `<h3>${cliente.cliente}</h3>`;
    html += `<p><b>Classe:</b> ${cliente.classe}</p>`;
    html += `<p><b>Risco:</b> ${(cliente.probabilidade * 100).toFixed(0)}%</p>`;

    if (cliente.motivos.length) {
        html += "<b>Motivos:</b><ul>";
        cliente.motivos.forEach(m => html += `<li>${m}</li>`);
        html += "</ul>";
    }

    document.getElementById("detalhes").innerHTML = html;
}
