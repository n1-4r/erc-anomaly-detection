from setuptools import find_packages, setup

package_name = 'anomaly_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smartnihar',
    maintainer_email='niharmahadevb@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'anomaly_detection_node = anomaly_detection.anomaly_detection_node:main',
            'anomaly_deduplication_node = anomaly_detection.anomaly_deduplicator_node:main',
            'image_monitor_node = anomaly_detection.image_monitor_node:main',
            'pdf_generator_node = anomaly_detection.pdf_generator:main',
        ],
    },
)
