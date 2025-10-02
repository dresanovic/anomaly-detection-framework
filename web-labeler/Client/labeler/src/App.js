import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-moment';
import { Chart, registerables } from 'chart.js';
import './App.css';

Chart.register(...registerables);

// Configurable column names
const CONFIG = {
    timeColumn: "Time", // Change this to your time column name if needed
    energyColumn: "Inv 1 AC-Leistung (W)" // Change this to your energy column name if needed
};

function App() {
    const [file, setFile] = useState(null);
    const [data, setData] = useState([]);
    // State to handle inline editing for the energy column
    const [editingRowIndex, setEditingRowIndex] = useState(null);
    const [editingValue, setEditingValue] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const uploadFile = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            await axios.post('http://localhost:8000/upload', formData);
            fetchData();
        } catch (error) {
            console.error("Upload error", error);
        }
    };

    const fetchData = async () => {
        try {
            const response = await axios.get('http://localhost:8000/data');
            setData(response.data);
        } catch (error) {
            console.error("Data fetch error", error);
        }
    };

    const toggleAnomaly = async (row) => {
        const newLabel = row.anomaly_label === 1 ? -1 : 1;
        try {
            await axios.post('http://localhost:8000/update_anomaly', null, {
                params: { time: row[CONFIG.timeColumn], anomaly_label: newLabel }
            });
            fetchData();
        } catch (error) {
            console.error("Update anomaly error", error);
        }
    };

    // Function to handle change in the energy value input field
    const handleEditChange = (e) => {
        setEditingValue(e.target.value);
    };

    // Function to submit the updated energy value when editing is complete
    const handleEditBlur = async (row, index) => {
        try {
            await axios.post('http://localhost:8000/update_energy', null, {
                params: { time: row[CONFIG.timeColumn], new_value: parseFloat(editingValue) }
            });
            setEditingRowIndex(null);
            fetchData();
        } catch (error) {
            console.error("Update energy error", error);
        }
    };

    // Prepare chart data using the configurable column names
    const chartData = {
        labels: data.map(row => row[CONFIG.timeColumn]),
        datasets: [
            {
                label: CONFIG.energyColumn,
                data: data.map(row => row[CONFIG.energyColumn]),
                borderColor: "blue",
                fill: false,
                tension: 0.1
            },
            {
                label: "Anomalies",
                data: data.map(row =>
                    row.anomaly_label === -1 ? row[CONFIG.energyColumn] : null
                ),
                borderColor: "red",
                backgroundColor: "red",
                pointRadius: 5,
                showLine: false,
            }
        ]
    };

    const chartOptions = {
        scales: {
            x: {
                type: 'time',
                time: { parser: 'YYYY-MM-DD', tooltipFormat: 'll' },
                title: { display: true, text: CONFIG.timeColumn }
            },
            y: { title: { display: true, text: CONFIG.energyColumn } }
        },
        onClick: (evt, elements) => {
            if (elements && elements.length > 0) {
                const index = elements[0].index;
                const row = data[index];
                toggleAnomaly(row);
            }
        }
    };

    // Utility to convert JSON data to CSV (unchanged)
    const convertToCSV = (objArray) => {
        if (!objArray || !objArray.length) return '';
        const keys = Object.keys(objArray[0]);
        const header = keys.join(',');
        const csvRows = objArray.map(row =>
            keys.map(k => JSON.stringify(row[k], replacer)).join(',')
        );
        return [header, ...csvRows].join('\r\n');
    };

    const replacer = (key, value) => (value === null ? '' : value);

    const handleSave = () => {
        if (!data || !data.length) {
            console.error("No data to save");
            return;
        }
        const csv = convertToCSV(data);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", "processed_data.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    useEffect(() => {
        fetchData();
    }, []);

    return (
        <div style={{ padding: "20px" }}>
            <h1>CSV Upload and Anomaly Visualization</h1>
            <div style={{ marginBottom: "10px" }}>
                <input type="file" accept=".csv" onChange={handleFileChange} />
                <button onClick={uploadFile} style={{ marginLeft: "10px" }}>
                    Upload CSV
                </button>
                <button onClick={handleSave} style={{ marginLeft: "10px" }}>
                    Save Data
                </button>
            </div>
            <div className="content-container">
                <div className="table-container">
                    <h2>Data Table</h2>
                    <table border="1" cellPadding="5" style={{ borderCollapse: "collapse", width: "100%" }}>
                        <thead>
                        <tr>
                            {data[0] && Object.keys(data[0]).map((col) => (
                                <th key={col}>{col}</th>
                            ))}
                            <th>Action</th>
                        </tr>
                        </thead>
                        <tbody>
                        {data.map((row, idx) => (
                            <tr key={idx}>
                                {Object.keys(row).map((col) => (
                                    <td key={col}>
                                        {col === CONFIG.energyColumn ? (
                                            editingRowIndex === idx ? (
                                                <input
                                                    type="number"
                                                    value={editingValue}
                                                    onChange={handleEditChange}
                                                    onBlur={() => handleEditBlur(row, idx)}
                                                    autoFocus
                                                />
                                            ) : (
                                                <span
                                                    onClick={() => {
                                                        setEditingRowIndex(idx);
                                                        setEditingValue(row[col]);
                                                    }}
                                                    style={{ cursor: "pointer" }}
                                                >
                                                    {row[col]}
                                                </span>
                                            )
                                        ) : (
                                            row[col]
                                        )}
                                    </td>
                                ))}
                                <td>
                                    <button onClick={() => toggleAnomaly(row)}>
                                        {row.anomaly_label === 1 ? "Mark as Anomaly" : "Unmark Anomaly"}
                                    </button>
                                </td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
                <div className="chart-container">
                    <h2>Time Series Plot</h2>
                    <Line data={chartData} options={chartOptions} />
                </div>
            </div>
        </div>
    );
}

export default App;
