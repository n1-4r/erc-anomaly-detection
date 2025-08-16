import rclpy
from rclpy.node import Node
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ollama import AsyncClient
import csv
import asyncio
import os


# Initialize CSV file with headers
file = open("/path/to/solid-memory/data.csv", 'w')
writer = csv.writer(file)
writer.writerow(['Path to image', 'location', 'object_identified', 'description'])
file.close()


class FileWatchNode(Node):
    def __init__(self):
        super().__init__('file_watch_node')
        self.watchDirectory = "/path/to/ProcessedImages"
        self.observer = Observer()
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive=True)
        self.observer.start()
        self.get_logger().info(f"Started watching {self.watchDirectory}")

    def destroy_node(self):
        self.observer.stop()
        self.observer.join()
        super().destroy_node()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            source_path = event.src_path
            print("file created and processing")
            file = open("/path/to/solid-memory/data.csv", 'a')
            writer = csv.writer(file)

            async def chat():
                content = ""
                message = [
                    {
                        'role': 'user',
                        'content': "Describe the given object in the bounding box.",
                        'images': [source_path]
                    },
                    {
                        'role': 'system',
                        'content': """You are an image analysis AI specializing in detecting anomalies in Martian surface images.
TASK:
1. Examine the provided image and look for anything that might be unusual, artificial, or inconsistent with typical Martian terrain.
2. Focus on:
   - Artificial-looking objects or materials
   - Unusual shapes, patterns, colors, or textures
   - Structures, debris, or objects that appear manufactured
   - Natural elements that appear altered, damaged, or out of place
3. Avoid describing ordinary rocks, sand, dust, and common geological features unless they have unusual properties or patterns.
4. If nothing seems unusual, respond with:
   No clear anomaly detected.
5. If anomalies are present:
   - Write 1–2 sentences (30–50 words) describing them.
   - Mention why they might be unusual in the Martian context.
   - It’s acceptable to note that you are unsure, but avoid wild speculation.

EXAMPLES:
- Input: Image contains only rocks and dust.
  Output: No clear anomaly detected.

- Input: Image shows a metallic fragment partly buried.
  Output: A small, reflective metallic piece protrudes from the sand. Its smooth surfaces and sharp angles stand out from the irregular shapes of surrounding rocks.

- Input: Image shows a rock with a perfect circular hole.
  Output: A dark rock with a precise circular hole is visible. The regularity of the hole is uncommon in natural erosion patterns.
  KEEP IN MIND THAT YOU NEED TO LOOK AT IT FROM A MARS PERSPECTIVE.
"""
                    }
                ]
                async for part in await AsyncClient().chat(
                    model='qwen2.5vl:7b-q8_0',
                    messages=message,
                    stream=True
                ):
                    content += part['message']['content']
                print(content)

                first_word = content.split()[0] if content else "None"

                writer.writerow([source_path, ' ', first_word, content.replace(first_word, "", 1)])
                file.close()

            asyncio.run(chat())


def main(args=None):
    rclpy.init(args=args)
    node = FileWatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
