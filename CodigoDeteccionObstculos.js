function CollisionBox(x, y, w, h) {
    this.x = x;
    this.y = y;
    this.width = w;
    this.height = h;
}

function compareBox(tRexBox, obstacleBox) {
    let crashed = false;
    const tRexBoxX = tRexBox.x;
    const tRexBoxY = tRexBox.y;

    const obstacleBoxX = obstacleBox.x;
    const obstacleBoxY = obstacleBox.y;

    // Axis-Aligned Bounding Box method.
    if (tRexBox.x < obstacleBoxX + obstacleBox.width &&
        tRexBox.x + tRexBox.width > obstacleBoxX &&
        tRexBox.y < obstacleBox.y + obstacleBox.height &&
        tRexBox.height + tRexBox.y > obstacleBox.y) {
        crashed = true;
    }

    return crashed;
}


function checkCollision(obstacles, tRex) {
    if (obstacles.length == 0) {
        return false;
    }

    const obstacle = obstacles[0]

    const tRexBox = new CollisionBox(
        tRex.xPos + 1,
        tRex.yPos + 1,
        tRex.config.WIDTH - 2,
        tRex.config.HEIGHT - 2);

    const obstacleBox = new CollisionBox(
        obstacle.xPos + 1,
        obstacle.yPos + 1,
        obstacle.typeConfig.width * obstacle.size - 2,
        obstacle.typeConfig.height - 2);


    // Simple outer bounds check.
    return compareBox(tRexBox, obstacleBox)
}

return checkCollision(Runner.instance_.horizon.obstacles, Runner.instance_.tRex);