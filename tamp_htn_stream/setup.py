from setuptools import setup
import glob

package_name = "tamp_htn_stream"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/data", glob.glob("data/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Core package: HTN-style TAMP stream planner runner with plugin loading.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "planner = tamp_htn_stream.planner:main",
            "domain_visualization = tamp_htn_stream.domain_visualization:main",
            "summarize_tests = tamp_htn_stream.summarize_tests:main",
        ],
    },
)
