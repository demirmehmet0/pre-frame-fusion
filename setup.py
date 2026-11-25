import setuptools
setuptools.setup(
    name="pre-frame-fusion",
    version="0.1.0",
    author="DigiNova",
    author_email='info@diginova.com.tr',
    description="Frame Fusion - Preprocessing Component for NOVAVISION",
    url='https://github.com/novavision-ai/pre-frame-fusion',
    license='MIT',
    install_requires=['sdk', 'opencv-python-headless', 'numpy'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=[
        'pre.frame_fusion',
        'pre.frame_fusion.executors',
        'pre.frame_fusion.models',
        'pre.frame_fusion.utils'
    ],
    package_dir={'pre.frame_fusion': 'src'},
    python_requires=">=3.6"
)
