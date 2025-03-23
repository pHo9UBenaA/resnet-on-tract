import { Hono } from 'hono'
import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static'
import { readFileSync } from 'fs'

const app = new Hono()

app.get('/', (c) => {
    return c.html(
        readFileSync('./index.html', 'utf-8')
    )
})

app.use('/*', serveStatic({ root: './' }))

// 画像をバイパスする
app.get('/image', async (c) => {
  const url = c.req.query('url')

  if (!url) {
    return c.text('url is required', 400)
  }

  try {
    const response = await fetch(url)

    if (!response.ok) {
      return c.text('File not found', 404)
    }

    const arrayBuffer = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/jpeg'

    return c.body(arrayBuffer, { headers: { 'Content-Type': contentType } })
  } catch (error) {
    return c.text('File not found', 404)
  }
})

serve(app, (info) => {
  console.log(`Server is running on http://localhost:${info.port}`)
})
