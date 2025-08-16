import rclpy
from rclpy.node import Node
import csv
from weasyprint import HTML

class CsvToHtmlPdfNode(Node):
    def __init__(self):
        super().__init__('csv_to_html_pdf_node')
        self.get_logger().info("CSV to HTML/PDF Node started.")

        files = []
        description = {}
        file = open("/path/to/data.csv", 'r')
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            file_name = row[0]
            files.append(file_name)
            description[row[0]] = row[3]
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #eef2f3, #dfe9f3);
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            font-size: 2.2em;
            margin-bottom: 15px;
            letter-spacing: 1px;
        }}
        .stats {{
            max-width: 500px;
            margin: 0 auto 25px auto;
            padding: 15px;
            background: #ffffffcc;
            border-radius: 12px;
            text-align: center;
            font-size: 1.1em;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
            padding: 10px;
        }}
        .image-card {{
            background: white;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .image-card:hover {{
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        .filename {{
            padding: 12px;
            font-weight: bold;
            background: #f7f9fa;
            font-size: 0.95em;
            color: #4a4a4a;
            border-bottom: 1px solid #ececec;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .description {{
            padding: 12px;
            font-size: 0.9em;
            color: #555;
            background: #fafbfc;
            border-top: 1px solid #f0f0f0;
            min-height: 50px;
        }}
        @media (max-width: 600px) {{
            body {{
                padding: 10px;
            }}
            h1 {{
                font-size: 1.6em;
            }}
        }}
    </style>
</head>
<body>
    <h1>Image Processing Results</h1>
    <div class="stats">
        📊 Total images processed: {len(files)}
    </div>
    <div class="gallery">
"""

        # Add image cards
        for file_name in files:
            html_content += f"""
        <div class="image-card">
            <img src="{file_name}" alt="{file_name}">
            <div class="description">{description.get(file_name, 'No description available')}</div>
        </div>
    """

        # Close HTML
        html_content += """
    </div>
</body>
</html>
"""

        # Save HTML file
        html_file_path = '/path/for/results.html'
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.get_logger().info(f"HTML report generated at: {html_file_path}")

        HTML(html_file_path).write_pdf("/path/for/report.pdf")
        self.get_logger().info("PDF report generated: report.pdf")


def main(args=None):
    rclpy.init(args=args)
    node = CsvToHtmlPdfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


