from setuptools import find_packages, setup

package_name = 'control'

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
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'up = control.up:main',
            'up_forward = control.up_forward:main',
            'up_forward_align = control.up_forward_align:main',
            'test_align = control.test_align:main',
            'upv12 = control.upv12:main',
            'upv12noplot = control.upv12noplot:main',
        ],
    },
)
