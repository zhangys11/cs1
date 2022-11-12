function createSpectrumChart(id, data, chart_title, chart_subtitle,
    xAxis_unit = '', yAxis_unit = '', xAxis_title = '', yAxis_title = 'Intensity')
{
    // create the chart
    Highcharts.chart(id, {
        exporting: {
            chartOptions: { // specific options for the exported image
                plotOptions: {
                    series: {
                        dataLabels: {
                            enabled: false
                        }
                    }
                }
            },
            fallbackToExportServer: false
        },

        chart: {
            type: 'area',
            zoomType: 'x',
            panning: true,
            panKey: 'shift'
        },

        title: {
            text: chart_title
        },

        subtitle: {
            text: chart_subtitle
        },

        xAxis: {
            labels: {
                format: '{value} ' + xAxis_unit
            },
            minRange: 5,
            title: {
                text: xAxis_title
            }
        },

        yAxis: {
            startOnTick: true,
            endOnTick: false,
            maxPadding: 0.35,
            title: {
                text: yAxis_title
            },
            labels: {
                format: '{value} ' + yAxis_unit
            }
        },

        tooltip: {
            headerFormat: (xAxis_title ? xAxis_title + ': ' : '') /*conditional (ternary) operator*/
                + '{point.x:.1f} ' + xAxis_unit + '<br>',
            pointFormat: '{point.y} ' + yAxis_unit,
            shared: true
        },

        legend: {
            enabled: false
        },

        plotOptions: {
            area: {
                fillColor: {
                    linearGradient: {
                        x1: 0,
                        y1: 0,
                        x2: 0,
                        y2: 1
                    },
                    stops: [
                        [0, Highcharts.getOptions().colors[0]],
                        [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                    ]
                },
                marker: {
                    radius: 2
                },
                lineWidth: 1,
                states: {
                    hover: {
                        lineWidth: 1
                    }
                },
                threshold: null
            },
        },

        series: [{
            data: data,
            lineColor: Highcharts.getOptions().colors[1],
            color: Highcharts.getOptions().colors[2],
            fillOpacity: 0.5,
            name: 'spectrum',
            marker: {
                enabled: false
            },
            threshold: null
        }]
    });
}

function createSpectraChart(id, data, chart_title, chart_subtitle,
    xAxis_unit = '', yAxis_unit = '', xAxis_title = '', yAxis_title = '') {
    // create the chart
    Highcharts.chart(id, {
        exporting: {
            chartOptions: { // specific options for the exported image
                plotOptions: {
                    series: {
                        dataLabels: {
                            enabled: false
                        }
                    }
                }
            },
            fallbackToExportServer: false
        },

        chart: {
            type: 'line',
            zoomType: 'x',
            panning: true,
            panKey: 'shift'
        },

        title: {
            text: chart_title
        },

        //subtitle: {
        //    text: chart_subtitle
        //},

        xAxis: {
            labels: {
                format: '{value} ' + xAxis_unit
            },
            minRange: 5,
            title: {
                text: xAxis_title
            }
        },

        yAxis: {
            startOnTick: true,
            endOnTick: false,
            maxPadding: 0.35,
            title: {
                text: yAxis_title
            },
            labels: {
                format: '{value} ' + yAxis_unit
            }
        },

        tooltip: {
            headerFormat: (xAxis_title ? xAxis_title + ': ' : '') /*conditional (ternary) operator*/
                + '{point.x:.1f} ' + xAxis_unit + '<br>',
            pointFormat: yAxis_title +': {point.y} ' + yAxis_unit,
            shared: true
        },

        legend: {
            enabled: true,
            align: 'right',
            // verticalAlign: 'top',
            layout: 'vertical',
        },

        series: data
    });
}
