English| [简体中文](./README_cn.md)

# Environment Initialization

```
$ yarn
```

This command will install the necessary plugins.

# Development

```
$ yarn start
```

This command will start a local development server and open a browser window. Most changes are reflected in real-time without the need to restart the server.

# Compilation

```
$ yarn build
```

This command will generate static content in the "build" directory, which can be used with any static content hosting service. After compilation, it is not supported to directly open .html files for viewing. Use `npm run serve` to start the service.

# Error Handling
1. If the environment initialization using `yarn` fails, try upgrading yarn to the latest version
```
npm install yarn@latest -g
```
2. If starting the local development server with `yarn start` fails, try upgrading the nodejs version
```
# Install the n module
sudo npm install -g n

# Upgrade nodejs to the specified version
sudo n node_version_number

# Check the nodejs version (need to restart the terminal)
node -v
```